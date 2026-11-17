// SSD tier: O_DIRECT batched reads of archive records into caller-owned pinned memory.
//
// The NVMe archive writes records through its buffered descriptor. Readers that
// assemble a selection (QSA window assembly) want the selected records in a
// pinned, page-aligned staging slot as fast as the drive allows, with no page
// cache copy and no bounce buffer: the slot is then copied to the GPU by DMA.
//
// The reader reopens the archive's inode through /proc/self/fd with O_DIRECT, so
// it also works for the unlinked (ephemeral) serving arena. A batch lists record
// slots in destination order. Runs of consecutive slots that land contiguously in
// the destination are merged, then split into requests of at most
// max_request_bytes and served by a pool of pread threads (thread count = queue
// depth). Linux writes back dirty page-cache pages of a range before an O_DIRECT
// read of it, so records the buffered descriptor has written are read coherently.
#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace vllm_kvmem::direct_reads {

constexpr uint64_t kAlignment = 4096;

struct BatchResult {
    uint64_t bytes = 0;
    uint32_t requests = 0;
    uint64_t wall_ns = 0;
};

class DirectRecordReader {
public:
    DirectRecordReader(int source_fd, uint64_t slot_bytes, uint32_t slot_count, int threads,
                       uint64_t max_request_bytes)
        : slot_bytes_(slot_bytes), slot_count_(slot_count), max_request_(max_request_bytes) {
        if (source_fd < 0) throw std::invalid_argument("Direct reads need an open archive descriptor");
        if (slot_bytes == 0 || slot_bytes % kAlignment)
            throw std::invalid_argument("Direct reads need records that are a multiple of 4 KiB");
        if (threads < 1 || threads > 64) throw std::invalid_argument("Direct read threads must be in [1, 64]");
        if (max_request_bytes < kAlignment || max_request_bytes % kAlignment)
            throw std::invalid_argument("Direct read request size must be a positive multiple of 4 KiB");
        const std::string proc = "/proc/self/fd/" + std::to_string(source_fd);
        fd_ = ::open(proc.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd_ < 0)
            throw std::system_error(errno, std::generic_category(), "O_DIRECT reopen of the archive failed");
        workers_.reserve(threads);
        for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { run(); });
    }

    ~DirectRecordReader() { close(); }
    DirectRecordReader(const DirectRecordReader &) = delete;
    DirectRecordReader &operator=(const DirectRecordReader &) = delete;

    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_) return;
            stopping_ = true;
        }
        work_cv_.notify_all();
        for (auto &worker : workers_) worker.join();
        workers_.clear();
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
    }

    // Queue a batch: slots[i] lands at destination + i * slot_bytes. A negative
    // slot leaves its destination record untouched (the caller fills it). A hole
    // [hole_offset, hole_offset + hole_bytes) inside every record is not read
    // (bytes no reader of this batch uses); the rest of a run of consecutive
    // slots is read as few large requests.
    uint64_t submit(const std::vector<int32_t> &slots, uint64_t destination, uint64_t destination_bytes,
                    uint64_t hole_offset = 0, uint64_t hole_bytes = 0) {
        if (slots.empty()) throw std::invalid_argument("Empty direct read batch");
        if (destination == 0 || destination % kAlignment)
            throw std::invalid_argument("Direct read destination must be 4 KiB aligned");
        if (destination_bytes < slots.size() * slot_bytes_)
            throw std::invalid_argument("Direct read destination is smaller than the batch");
        if (hole_offset % kAlignment || hole_bytes % kAlignment || hole_offset + hole_bytes > slot_bytes_ ||
                hole_bytes == slot_bytes_)
            throw std::invalid_argument("Direct read hole must be a 4 KiB aligned part of a record");
        for (int32_t slot : slots)
            if (slot >= 0 && static_cast<uint32_t>(slot) >= slot_count_)
                throw std::out_of_range("Direct read slot outside the archive");
        auto batch = std::make_shared<Batch>();
        batch->started = std::chrono::steady_clock::now();
        std::vector<Job> jobs;
        auto emit = [&](uint64_t offset, uint8_t *target, uint64_t remaining) {
            while (remaining) {
                const uint64_t bytes = std::min(remaining, max_request_);
                jobs.push_back(Job{batch, offset, target, bytes});
                offset += bytes;
                target += bytes;
                remaining -= bytes;
            }
        };
        size_t i = 0;
        while (i < slots.size()) {
            if (slots[i] < 0) {
                ++i;
                continue;
            }
            size_t j = i + 1;
            while (j < slots.size() && slots[j] == slots[j - 1] + 1) ++j;
            // File and destination are both record-strided over the run.
            const uint64_t file = static_cast<uint64_t>(slots[i]) * slot_bytes_;
            auto *target = reinterpret_cast<uint8_t *>(destination + i * slot_bytes_);
            const uint64_t length = (j - i) * slot_bytes_;
            uint64_t cursor = 0;
            if (hole_bytes) {
                for (size_t k = 0; k < j - i; ++k) {
                    const uint64_t hole = k * slot_bytes_ + hole_offset;
                    if (hole > cursor) emit(file + cursor, target + cursor, hole - cursor);
                    cursor = hole + hole_bytes;
                }
            }
            if (length > cursor) emit(file + cursor, target + cursor, length - cursor);
            i = j;
        }
        return enqueue(batch, jobs);
    }

    // Queue per-record plane slices: for plane p = (record_offset, bytes, destination_offset)
    // and record i, bytes at slots[i] * slot_bytes + record_offset land at
    // destination + destination_offset + i * bytes (plane-major destination).
    // Used by per-layer selections, which need one layer's slices of each record.
    uint64_t submit_planes(const std::vector<int32_t> &slots,
                           const std::vector<std::array<uint64_t, 3>> &planes,
                           uint64_t destination, uint64_t destination_bytes) {
        if (slots.empty() || planes.empty()) throw std::invalid_argument("Empty direct plane batch");
        if (destination == 0 || destination % kAlignment)
            throw std::invalid_argument("Direct read destination must be 4 KiB aligned");
        for (const auto &plane : planes) {
            const uint64_t offset = plane[0], bytes = plane[1], target = plane[2];
            if (bytes == 0 || offset % kAlignment || bytes % kAlignment || target % kAlignment ||
                    offset + bytes > slot_bytes_)
                throw std::invalid_argument("Direct read planes must be 4 KiB aligned parts of a record");
            if (target + slots.size() * bytes > destination_bytes)
                throw std::invalid_argument("Direct read plane exceeds the destination");
        }
        for (int32_t slot : slots)
            if (slot >= 0 && static_cast<uint32_t>(slot) >= slot_count_)
                throw std::out_of_range("Direct read slot outside the archive");
        auto batch = std::make_shared<Batch>();
        batch->started = std::chrono::steady_clock::now();
        std::vector<Job> jobs;
        jobs.reserve(slots.size() * planes.size());
        for (const auto &plane : planes) {
            for (size_t i = 0; i < slots.size(); ++i) {
                if (slots[i] < 0) continue;
                auto *target = reinterpret_cast<uint8_t *>(destination + plane[2] + i * plane[1]);
                const uint64_t file = static_cast<uint64_t>(slots[i]) * slot_bytes_ + plane[0];
                // Consecutive slots of a whole-record plane are contiguous in both file and destination.
                if (!jobs.empty() && plane[1] == slot_bytes_ && jobs.back().offset + jobs.back().bytes == file &&
                        jobs.back().target + jobs.back().bytes == target && jobs.back().bytes < max_request_) {
                    jobs.back().bytes += plane[1];
                    continue;
                }
                jobs.push_back(Job{batch, file, target, plane[1]});
            }
        }
        return enqueue(batch, jobs);
    }

    // Queue raw extents {file_offset, bytes, destination_offset}, all 4 KiB aligned.
    uint64_t submit_extents(const std::vector<std::array<uint64_t, 3>> &extents, uint64_t destination,
                            uint64_t destination_bytes) {
        if (destination == 0 || destination % kAlignment)
            throw std::invalid_argument("Direct read destination must be 4 KiB aligned");
        const uint64_t file_bytes = static_cast<uint64_t>(slot_count_) * slot_bytes_;
        auto batch = std::make_shared<Batch>();
        batch->started = std::chrono::steady_clock::now();
        std::vector<Job> jobs;
        for (const auto &extent : extents) {
            uint64_t file = extent[0], remaining = extent[1], target = extent[2];
            if (file % kAlignment || remaining % kAlignment || target % kAlignment ||
                    file + remaining > file_bytes || target + remaining > destination_bytes)
                throw std::invalid_argument("Direct read extent is unaligned or out of range");
            while (remaining) {
                const uint64_t bytes = std::min(remaining, max_request_);
                jobs.push_back(Job{batch, file, reinterpret_cast<uint8_t *>(destination + target), bytes});
                file += bytes;
                target += bytes;
                remaining -= bytes;
            }
        }
        return enqueue(batch, jobs);
    }

    // Block until the batch is complete; rethrows the first I/O error.
    BatchResult wait(uint64_t ticket) {
        std::unique_lock<std::mutex> lock(mu_);
        auto found = batches_.find(ticket);
        if (found == batches_.end()) throw std::invalid_argument("Unknown direct read ticket");
        auto batch = found->second;
        done_cv_.wait(lock, [&] { return batch->pending == 0; });
        batches_.erase(found);
        if (batch->error) throw std::system_error(batch->error, std::generic_category(), batch->message);
        BatchResult result;
        result.bytes = batch->bytes;
        result.requests = batch->requests;
        result.wall_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            batch->finished - batch->started).count());
        return result;
    }

    bool done(uint64_t ticket) {
        std::lock_guard<std::mutex> lock(mu_);
        auto found = batches_.find(ticket);
        if (found == batches_.end()) throw std::invalid_argument("Unknown direct read ticket");
        return found->second->pending == 0;
    }

    int threads() const { return static_cast<int>(workers_.size()); }

private:
    struct Batch {
        size_t pending = 0;
        uint32_t requests = 0;
        uint64_t bytes = 0;
        int error = 0;
        std::string message;
        std::chrono::steady_clock::time_point started, finished;
    };
    struct Job {
        std::shared_ptr<Batch> batch;
        uint64_t offset;
        uint8_t *target;
        uint64_t bytes;
    };

    uint64_t enqueue(const std::shared_ptr<Batch> &batch, std::vector<Job> &jobs) {
        batch->pending = jobs.size();
        batch->requests = static_cast<uint32_t>(jobs.size());
        if (jobs.empty()) batch->finished = batch->started;
        uint64_t ticket;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_) throw std::runtime_error("Direct record reader is closed");
            ticket = ++next_ticket_;
            batches_[ticket] = batch;
            for (auto &job : jobs) queue_.push_back(std::move(job));
        }
        work_cv_.notify_all();
        return ticket;
    }

    void run() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mu_);
                work_cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) return;  // stopping, nothing left
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            int error = 0;
            std::string message;
            uint64_t done = 0;
            while (done < job.bytes) {
                const ssize_t n = ::pread(fd_, job.target + done, job.bytes - done,
                                          static_cast<off_t>(job.offset + done));
                if (n < 0) {
                    if (errno == EINTR) continue;
                    error = errno;
                    message = "Direct archive read failed";
                    break;
                }
                if (n == 0) {
                    error = EIO;
                    message = "Direct archive read hit the end of the arena (record not on disk)";
                    break;
                }
                done += static_cast<uint64_t>(n);
            }
            {
                std::lock_guard<std::mutex> lock(mu_);
                job.batch->bytes += done;
                if (error && !job.batch->error) {
                    job.batch->error = error;
                    job.batch->message = message;
                }
                if (--job.batch->pending == 0) {
                    job.batch->finished = std::chrono::steady_clock::now();
                    done_cv_.notify_all();
                }
            }
        }
    }

    uint64_t slot_bytes_;
    uint32_t slot_count_;
    uint64_t max_request_;
    int fd_ = -1;
    std::mutex mu_;
    std::condition_variable work_cv_, done_cv_;
    std::deque<Job> queue_;
    std::unordered_map<uint64_t, std::shared_ptr<Batch>> batches_;
    uint64_t next_ticket_ = 0;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
};

}  // namespace vllm_kvmem::direct_reads
