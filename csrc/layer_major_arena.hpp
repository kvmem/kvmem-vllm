// SSD tier: a layer-major, unpadded NVMe arena for archive records.
//
// An archive record is [raw plane][values plane] for every archive layer of one
// block. Per-layer selections (KVMem's layerwise Query) need one layer's slice
// of many records; in the record layout those slices are 16 KiB pieces spread
// over the file, which an NVMe drive serves at a fraction of its bandwidth.
//
// This arena stores each record as G groups (one per model layer). A group is a
// list of segments of the record (offset, rows x row_bytes at row_stride), packed
// back to back; group g of slot s lives at group_base[g] + s * group_bytes[g], so
// a layer's selected blocks are long contiguous runs when their slots are. Bytes
// of the record outside every segment are not stored: write() checks they are
// zero (padding) and read() returns them as zero, so the layout is lossless.
//
// All I/O uses O_DIRECT on an unlinked file. Whole-record write()/read() are
// synchronous (archive worker threads, GIL released by the binding); group reads
// for direct assembly are asynchronous through the DirectRecordReader pool.
#pragma once

#include <cstdlib>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "direct_record_reader.hpp"

namespace vllm_kvmem::direct_reads {

struct Segment {
    uint64_t offset;      // in the record
    uint32_t rows;
    uint32_t row_bytes;
    uint32_t row_stride;  // record bytes between rows (ignored when rows == 1)
};

class LayerMajorArena {
public:
    LayerMajorArena(const std::string &directory, uint32_t slots, uint64_t record_bytes,
                    std::vector<std::vector<Segment>> groups, int threads, uint64_t max_request_bytes)
        : slots_(slots), record_bytes_(record_bytes), groups_(std::move(groups)) {
        if (slots == 0 || record_bytes == 0 || groups_.empty())
            throw std::invalid_argument("Layer-major arena needs slots, records and groups");
        std::vector<uint8_t> covered(record_bytes, 0);
        uint64_t base = 0;
        for (const auto &group : groups_) {
            uint64_t bytes = 0;
            for (const auto &segment : group) {
                if (segment.rows == 0 || segment.row_bytes == 0 ||
                        (segment.rows > 1 && segment.row_stride < segment.row_bytes))
                    throw std::invalid_argument("Layer-major segment has no bytes or overlapping rows");
                const uint64_t last = segment.offset + uint64_t(segment.rows - 1) * segment.row_stride;
                if (last + segment.row_bytes > record_bytes)
                    throw std::invalid_argument("Layer-major segment exceeds the record");
                for (uint32_t r = 0; r < segment.rows; ++r) {
                    const uint64_t start = segment.offset + uint64_t(r) * segment.row_stride;
                    for (uint64_t b = start; b < start + segment.row_bytes; ++b) {
                        if (covered[b]) throw std::invalid_argument("Layer-major segments overlap");
                        covered[b] = 1;
                    }
                }
                bytes += uint64_t(segment.rows) * segment.row_bytes;
            }
            if (bytes == 0 || bytes % kAlignment)
                throw std::invalid_argument("Layer-major group bytes must be a positive multiple of 4 KiB");
            group_bytes_.push_back(bytes);
            group_base_.push_back(base);
            base += uint64_t(slots) * bytes;
        }
        file_bytes_ = base;
        for (uint64_t b = 0; b < record_bytes;) {        // dropped (padding) ranges
            if (covered[b]) { ++b; continue; }
            uint64_t e = b;
            while (e < record_bytes && !covered[e]) ++e;
            dropped_.push_back({b, e - b});
            b = e;
        }
        std::string pattern = directory + "/kvmem-layer-major-XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        const int fd = ::mkstemp(name.data());
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "Cannot create the layer-major arena");
        ::unlink(name.data());
        buffered_fd_ = fd;
        if (::ftruncate(fd, static_cast<off_t>(file_bytes_)) != 0) {
            const int error = errno;
            ::close(fd);
            buffered_fd_ = -1;
            throw std::system_error(error, std::generic_category(), "Cannot size the layer-major arena");
        }
        const std::string proc = "/proc/self/fd/" + std::to_string(fd);
        direct_fd_ = ::open(proc.c_str(), O_RDWR | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) {
            const int error = errno;
            ::close(fd);
            buffered_fd_ = -1;
            throw std::system_error(error, std::generic_category(), "Layer-major arena needs O_DIRECT");
        }
        reader_ = std::make_unique<DirectRecordReader>(buffered_fd_, kAlignment,
                                                       static_cast<uint32_t>(file_bytes_ / kAlignment),
                                                       threads, max_request_bytes);
    }

    ~LayerMajorArena() { close(); }
    LayerMajorArena(const LayerMajorArena &) = delete;
    LayerMajorArena &operator=(const LayerMajorArena &) = delete;

    void close() {
        if (reader_) reader_->close();
        reader_.reset();
        if (direct_fd_ >= 0) ::close(direct_fd_);
        if (buffered_fd_ >= 0) ::close(buffered_fd_);
        direct_fd_ = buffered_fd_ = -1;
    }

    // Store whole records (archive layout) at slots. Throws if a record has
    // non-zero bytes outside every segment (the layout would lose them).
    void write(const std::vector<int32_t> &slots, const uint8_t *source, uint64_t source_bytes) {
        check_slots(slots, source_bytes);
        for (size_t i = 0; i < slots.size(); ++i) {
            const uint8_t *record = source + i * record_bytes_;
            for (const auto &range : dropped_)
                if (!all_zero(record + range[0], range[1]))
                    throw std::runtime_error("Archive record has non-zero bytes outside its layer-major segments");
        }
        for (size_t g = 0; g < groups_.size(); ++g) {
            for_runs(slots, g, [&](size_t first, size_t count) {
                uint8_t *packed = scratch(count * group_bytes_[g]);
                for (size_t k = 0; k < count; ++k)
                    pack(g, source + (first + k) * record_bytes_, packed + k * group_bytes_[g]);
                io_all(true, packed, count * group_bytes_[g],
                       group_base_[g] + uint64_t(slots[first]) * group_bytes_[g]);
            });
        }
    }

    // Rebuild whole records (archive layout; dropped bytes are zero).
    void read(const std::vector<int32_t> &slots, uint8_t *destination, uint64_t destination_bytes) {
        check_slots(slots, destination_bytes);
        std::memset(destination, 0, slots.size() * record_bytes_);
        for (size_t g = 0; g < groups_.size(); ++g) {
            for_runs(slots, g, [&](size_t first, size_t count) {
                uint8_t *packed = scratch(count * group_bytes_[g]);
                io_all(false, packed, count * group_bytes_[g],
                       group_base_[g] + uint64_t(slots[first]) * group_bytes_[g]);
                for (size_t k = 0; k < count; ++k)
                    unpack(g, packed + k * group_bytes_[g], destination + (first + k) * record_bytes_);
            });
        }
    }

    // Asynchronous packed group reads for direct assembly: for the k-th requested
    // group g, slot[i]'s packed group lands at destination + offsets[k] + i * group_bytes[g].
    // Negative slots are skipped (the caller fills them).
    uint64_t submit_groups(const std::vector<int32_t> &slots, const std::vector<uint32_t> &groups,
                           const std::vector<uint64_t> &offsets, uint64_t destination, uint64_t destination_bytes) {
        if (groups.size() != offsets.size() || slots.empty() || groups.empty())
            throw std::invalid_argument("Layer-major group read needs one destination per group");
        for (int32_t slot : slots)
            if (slot >= 0 && static_cast<uint32_t>(slot) >= slots_)
                throw std::out_of_range("Layer-major slot outside the arena");
        std::vector<std::array<uint64_t, 3>> extents;
        for (size_t k = 0; k < groups.size(); ++k) {
            const uint32_t g = groups[k];
            if (g >= groups_.size()) throw std::out_of_range("Layer-major group out of range");
            const uint64_t bytes = group_bytes_[g];
            size_t i = 0;
            while (i < slots.size()) {
                if (slots[i] < 0) { ++i; continue; }
                size_t j = i + 1;
                while (j < slots.size() && slots[j] == slots[j - 1] + 1) ++j;
                extents.push_back({group_base_[g] + uint64_t(slots[i]) * bytes, (j - i) * bytes,
                                   offsets[k] + i * bytes});
                i = j;
            }
        }
        return reader_->submit_extents(extents, destination, destination_bytes);
    }

    BatchResult wait(uint64_t ticket) { return reader_->wait(ticket); }
    bool done(uint64_t ticket) { return reader_->done(ticket); }
    uint64_t group_bytes(size_t g) const { return group_bytes_.at(g); }
    size_t group_count() const { return groups_.size(); }
    uint64_t record_bytes() const { return record_bytes_; }
    uint64_t file_bytes() const { return file_bytes_; }
    uint64_t stored_bytes_per_record() const {
        uint64_t total = 0;
        for (uint64_t b : group_bytes_) total += b;
        return total;
    }

private:
    void check_slots(const std::vector<int32_t> &slots, uint64_t bytes) const {
        if (slots.empty() || bytes < slots.size() * record_bytes_)
            throw std::invalid_argument("Layer-major record I/O buffer is smaller than the batch");
        for (int32_t slot : slots)
            if (slot < 0 || static_cast<uint32_t>(slot) >= slots_)
                throw std::out_of_range("Layer-major slot outside the arena");
    }

    // Runs of consecutive slots (in buffer order), at most kMaxRun bytes of one group each.
    template <class F>
    void for_runs(const std::vector<int32_t> &slots, size_t g, F &&run) const {
        const size_t limit = std::max<size_t>(1, kMaxRun / group_bytes_[g]);
        size_t i = 0;
        while (i < slots.size()) {
            size_t j = i + 1;
            while (j < slots.size() && j - i < limit && slots[j] == slots[j - 1] + 1) ++j;
            run(i, j - i);
            i = j;
        }
    }

    void pack(size_t g, const uint8_t *record, uint8_t *out) const {
        for (const auto &segment : groups_[g]) {
            if (segment.rows == 1 || segment.row_stride == segment.row_bytes) {
                const size_t bytes = size_t(segment.rows) * segment.row_bytes;
                std::memcpy(out, record + segment.offset, bytes);
                out += bytes;
                continue;
            }
            for (uint32_t r = 0; r < segment.rows; ++r, out += segment.row_bytes)
                std::memcpy(out, record + segment.offset + uint64_t(r) * segment.row_stride, segment.row_bytes);
        }
    }

    void unpack(size_t g, const uint8_t *in, uint8_t *record) const {
        for (const auto &segment : groups_[g]) {
            if (segment.rows == 1 || segment.row_stride == segment.row_bytes) {
                const size_t bytes = size_t(segment.rows) * segment.row_bytes;
                std::memcpy(record + segment.offset, in, bytes);
                in += bytes;
                continue;
            }
            for (uint32_t r = 0; r < segment.rows; ++r, in += segment.row_bytes)
                std::memcpy(record + segment.offset + uint64_t(r) * segment.row_stride, in, segment.row_bytes);
        }
    }

    static bool all_zero(const uint8_t *data, uint64_t bytes) {
        uint64_t acc = 0, i = 0;
        for (; i + 8 <= bytes; i += 8) {
            uint64_t word;
            std::memcpy(&word, data + i, 8);
            acc |= word;
        }
        for (; i < bytes; ++i) acc |= data[i];
        return acc == 0;
    }

    void io_all(bool write, uint8_t *data, uint64_t bytes, uint64_t offset) const {
        while (bytes) {
            const ssize_t n = write ? ::pwrite(direct_fd_, data, bytes, static_cast<off_t>(offset))
                                    : ::pread(direct_fd_, data, bytes, static_cast<off_t>(offset));
            if (n < 0) {
                if (errno == EINTR) continue;
                throw std::system_error(errno, std::generic_category(),
                                        write ? "Layer-major arena write failed" : "Layer-major arena read failed");
            }
            if (n == 0) throw std::runtime_error("Layer-major arena I/O made no progress");
            data += n;
            bytes -= static_cast<uint64_t>(n);
            offset += static_cast<uint64_t>(n);
        }
    }

    // Per-thread 4 KiB aligned packing buffer (archive workers run concurrently).
    static uint8_t *scratch(uint64_t bytes) {
        struct Buffer {
            void *data = nullptr;
            uint64_t size = 0;
            ~Buffer() { std::free(data); }
        };
        thread_local Buffer buffer;
        if (buffer.size < bytes) {
            std::free(buffer.data);
            buffer.data = nullptr;
            buffer.size = 0;
            if (::posix_memalign(&buffer.data, kAlignment, bytes) != 0)
                throw std::bad_alloc();
            buffer.size = bytes;
        }
        return static_cast<uint8_t *>(buffer.data);
    }

    static constexpr uint64_t kMaxRun = 8ull << 20;
    uint32_t slots_;
    uint64_t record_bytes_, file_bytes_ = 0;
    std::vector<std::vector<Segment>> groups_;
    std::vector<uint64_t> group_bytes_, group_base_;
    std::vector<std::array<uint64_t, 2>> dropped_;
    int buffered_fd_ = -1, direct_fd_ = -1;
    std::unique_ptr<DirectRecordReader> reader_;
};

}  // namespace vllm_kvmem::direct_reads
