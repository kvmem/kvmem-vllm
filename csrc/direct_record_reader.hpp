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
    // slot leaves its destination record untouched (the caller fills it).
    uint64_t submit(const std::vector<int32_t> &slots, uint64_t destination, uint64_t destination_bytes) {
        if (slots.empty()) throw std::invalid_argument("Empty direct read batch");
        if (destination == 0 || destination % kAlignment)
            throw std::invalid_argument("Direct read destination must be 4 KiB aligned");
        if (destination_bytes < slots.size() * slot_bytes_)
            throw std::invalid_argument("Direct read destination is smaller than the batch");
        for (int32_t slot : slots)
            if (slot >= 0 && static_cast<uint32_t>(slot) >= slot_count_)
                throw std::out_of_range("Direct read slot outside the archive");
        auto batch = std::make_shared<Batch>();
        batch->started = std::chrono::steady_clock::now();
        std::vector<Job> jobs;
        size_t i = 0;
        while (i < slots.size()) {
            if (slots[i] < 0) {
                ++i;
                continue;
            }
            size_t j = i + 1;
            while (j < slots.size() && slots[j] == slots[j - 1] + 1) ++j;
            uint64_t offset = static_cast<uint64_t>(slots[i]) * slot_bytes_;
            uint64_t remaining = (j - i) * slot_bytes_;
            auto *target = reinterpret_cast<uint8_t *>(destination + i * slot_bytes_);
            while (remaining) {
                const uint64_t bytes = std::min(remaining, max_request_);
                jobs.push_back(Job{batch, offset, target, bytes});
                offset += bytes;
                target += bytes;
                remaining -= bytes;
            }
            i = j;
        }
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
