#include "historical_payload.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <sys/mman.h>

namespace vllm_kvmem {
namespace {
// Control word: phase (bits 60-63), readers (bits 0-31), and for a shared
// payload (docs/prefix-sharing.md, phase 2) its extra owners beyond the
// publisher (bits 32-47) and how many of all owners are shared-prefix pool
// owners (bits 48-59). A single-owner payload has no owner bits, so its
// control words are exactly the values used before sharing existed.
constexpr unsigned shift = 60, owner_shift = 32, pool_shift = 48;
constexpr uint64_t refs_mask = (uint64_t{1} << owner_shift) - 1;
constexpr uint64_t owner_unit = uint64_t{1} << owner_shift, owner_max = (uint64_t{1} << 16) - 1;
constexpr uint64_t pool_unit = uint64_t{1} << pool_shift, pool_max = (uint64_t{1} << 12) - 1;
enum class Phase : uint64_t { Empty, Initializing, Live, Retired, Releasing, Spilled };
constexpr uint64_t control(Phase p, uint64_t refs = 0) { return (uint64_t(p) << shift) | refs; }
Phase phase(uint64_t value) { return Phase(value >> shift); }
uint64_t refs(uint64_t value) { return value & refs_mask; }
uint64_t extra_owners(uint64_t value) { return (value >> owner_shift) & owner_max; }
uint64_t pool_owners(uint64_t value) { return (value >> pool_shift) & pool_max; }
enum class DropKind { Plain, Pool, Unshared };
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "Historical payload lifetime requires lock-free 64-bit atomics");

// Opt-in fixed-size record pool. An archive grows monotonically while a long
// prompt is prefilled, so per-record heap allocations land on fresh pages and
// every publication pays first-touch faults (kernel zeroing) on the archive
// producer's critical path. The pool carves records from huge-page-advised
// slabs, pre-faults free records on its own thread and reuses released
// records LIFO. Only the memory location changes; every byte is copied as before.
class RecordPool {
public:
    RecordPool(size_t record_bytes, size_t chunk_mib, size_t ahead)
        : slot_bytes_((record_bytes + 4095) & ~size_t{4095}),
          chunk_bytes_(std::max(chunk_mib << 20, slot_bytes_ + huge_)),
          ahead_(ahead), filler_([this] { fill(); }) {}
    ~RecordPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        want_.notify_all();
        if (filler_.joinable()) filler_.join();
        for (auto &chunk : chunks_) munmap(chunk.first, chunk.second);
    }
    RecordPool(const RecordPool &) = delete;
    RecordPool &operator=(const RecordPool &) = delete;

    uint8_t *acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        uint8_t *record;
        if (!free_.empty()) {
            record = free_.back();
            free_.pop_back();
            hits_.fetch_add(1, std::memory_order_relaxed);
        } else {
            record = carve_locked();  // First touch happens in the caller's memcpy.
            misses_.fetch_add(1, std::memory_order_relaxed);
        }
        const bool low = free_.size() < ahead_;
        lock.unlock();
        if (low) want_.notify_one();
        return record;
    }
    void release(uint8_t *record) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        free_.push_back(record);  // vector capacity is reserved below; see carve_locked.
    }
    uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
    uint64_t misses() const { return misses_.load(std::memory_order_relaxed); }
    uint64_t reserved() const { return reserved_.load(std::memory_order_relaxed); }

private:
    static constexpr size_t huge_ = size_t{2} << 20;

    uint8_t *carve_locked() {
        if (!left_) {
#ifdef MAP_NORESERVE
            constexpr int reserve_flag = MAP_NORESERVE;
#else
            constexpr int reserve_flag = 0;
#endif
            void *raw = mmap(nullptr, chunk_bytes_ + huge_, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | reserve_flag, -1, 0);
            if (raw == MAP_FAILED) throw std::bad_alloc();
            chunks_.emplace_back(raw, chunk_bytes_ + huge_);
            auto aligned = (reinterpret_cast<uintptr_t>(raw) + huge_ - 1) & ~(huge_ - 1);
#ifdef MADV_HUGEPAGE
            // Linux transparent huge pages are a hint only; other platforms
            // use ordinary pages with identical contents and behaviour.
            madvise(reinterpret_cast<void *>(aligned), chunk_bytes_, MADV_HUGEPAGE);
#endif
            next_ = reinterpret_cast<uint8_t *>(aligned);
            left_ = chunk_bytes_ / slot_bytes_;
            // release() runs in a noexcept deleter: never let push_back allocate.
            free_.reserve(free_.size() + carved_ + left_);
            reserved_.fetch_add(chunk_bytes_ + huge_, std::memory_order_relaxed);
        }
        auto *record = next_;
        next_ += slot_bytes_;
        --left_;
        ++carved_;
        return record;
    }
    void fill() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_) {
            if (!ahead_ || free_.size() >= ahead_) {
                want_.wait_for(lock, std::chrono::milliseconds(50));
                continue;
            }
            uint8_t *batch[32];
            size_t n = 0;
            while (n < 32 && free_.size() + n < ahead_) batch[n++] = carve_locked();
            lock.unlock();
            for (size_t i = 0; i < n; ++i)  // Fault in every page off the publication path.
                for (size_t off = 0; off < slot_bytes_; off += 4096) batch[i][off] = 0;
            lock.lock();
            for (size_t i = 0; i < n; ++i) free_.push_back(batch[i]);
        }
    }

    const size_t slot_bytes_, chunk_bytes_, ahead_;
    std::mutex mutex_;
    std::condition_variable want_;
    std::vector<uint8_t *> free_;
    std::vector<std::pair<void *, size_t>> chunks_;
    uint8_t *next_ = nullptr;
    size_t left_ = 0, carved_ = 0;
    bool stop_ = false;
    std::atomic<uint64_t> hits_{0}, misses_{0}, reserved_{0};
    std::thread filler_;  // Last member: starts after every field above is ready.
};

struct PayloadFree {
    RecordPool *pool = nullptr;
    void operator()(uint8_t *record) const noexcept {
        if (pool) pool->release(record); else delete[] record;
    }
};
using Payload = std::unique_ptr<uint8_t[], PayloadFree>;
}

struct HistoricalPayloadArena::ExternalOwner::Node {
    void *data;
    void (*release)(void *);
    size_t bytes;
    std::shared_ptr<ExternalCompletion> completion = std::make_shared<ExternalCompletion>();
    Node *next = nullptr;
};

struct HistoricalPayloadArena::Shared {
    struct Entry {
        std::atomic<uint64_t> control{vllm_kvmem::control(Phase::Empty)};
        std::atomic<uint64_t> generation{0}, released_generation{0};
        Payload payload;
    };
    explicit Shared(Config c)
        : config(c),
          pool(c.pool_chunk_mib ? new RecordPool(c.record_bytes, c.pool_chunk_mib, c.pool_ahead_records)
                                : nullptr),
          entries(new Entry[c.capacity]), dirty_words((uint64_t(c.capacity) + 63) / 64),
          dirty(new std::atomic<uint64_t>[dirty_words]) {
        for (size_t i = 0; i < dirty_words; ++i) dirty[i].store(0, std::memory_order_relaxed);
    }
    Config config;
    // Declared before entries: payload deleters return records to a live pool.
    std::unique_ptr<RecordPool> pool;
    std::unique_ptr<Entry[]> entries;
    size_t dirty_words;
    std::unique_ptr<std::atomic<uint64_t>[]> dirty;
    std::atomic<bool> closing{false}, complete{false};
    std::atomic_flag reclaimer = ATOMIC_FLAG_INIT;
    std::atomic<uint64_t> used_slots{0}, used_bytes{0}, retired_bytes{0}, readers{0};
    std::atomic<uint64_t> released_slots{0}, released_bytes{0}, last_release_thread{0}, writers{0};
    std::atomic<uint64_t> external_owners{0}, external_bytes{0}, external_released_bytes{0};
    std::atomic<uint64_t> spilled_slots{0}, spilled_bytes{0};
    std::atomic<uint64_t> shared_slots{0}, pool_slots{0};
    std::atomic<ExternalOwner::Node *> external_pending{nullptr};
    ExternalOwner::Node *external_local = nullptr; // Sole reclaimer ownership.
    uint32_t cursor = 0; // Exclusive reclaimer ownership, not a foreground lock.
    size_t dirty_cursor = 0;
    std::mutex sleep_mutex;
    std::condition_variable wake;

    void mark_retired(uint32_t slot) {
        dirty[slot / 64].fetch_or(uint64_t{1} << (slot % 64), std::memory_order_release);
    }

    RetireResult drop(Handle h, DropKind kind);

    // Spilled -> Retired (no CPU bytes). Accounting is restored to the resident
    // form first so release_one's single accounting path stays balanced.
    bool retire_spilled(uint32_t slot) {
        auto &entry = entries[slot];
        auto expected = control(Phase::Spilled);
        if (!entry.control.compare_exchange_strong(expected, control(Phase::Retired),
                std::memory_order_acq_rel, std::memory_order_acquire)) return false;
        used_bytes.fetch_add(config.record_bytes, std::memory_order_relaxed);
        spilled_slots.fetch_sub(1, std::memory_order_relaxed);
        spilled_bytes.fetch_sub(config.record_bytes, std::memory_order_relaxed);
        retired_bytes.fetch_add(config.record_bytes, std::memory_order_relaxed);
        mark_retired(slot);
        return true;
    }

    bool release_one(uint32_t slot) {
        auto &entry = entries[slot];
        auto expected = control(Phase::Retired);
        if (!entry.control.compare_exchange_strong(expected, control(Phase::Releasing),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            if (phase(expected) == Phase::Retired) mark_retired(slot);
            return false;
        }
        const auto released_generation = entry.generation.load(std::memory_order_relaxed);
        entry.payload.reset(); // The only normal payload destruction site.
        used_bytes.fetch_sub(config.record_bytes, std::memory_order_relaxed);
        retired_bytes.fetch_sub(config.record_bytes, std::memory_order_relaxed);
        used_slots.fetch_sub(1, std::memory_order_relaxed);
        released_bytes.fetch_add(config.record_bytes, std::memory_order_relaxed);
        released_slots.fetch_add(1, std::memory_order_relaxed);
        last_release_thread.store(std::hash<std::thread::id>{}(std::this_thread::get_id()),
                                  std::memory_order_relaxed);
        entry.control.store(control(Phase::Empty), std::memory_order_release);
        // Completion follows the reusable state, including for a new producer
        // installing the next generation concurrently with this notification.
        entry.released_generation.store(released_generation, std::memory_order_release);
        return true;
    }

    size_t advance() {
        if (reclaimer.test_and_set(std::memory_order_acquire)) return 0;
        struct Unlock {
            std::atomic_flag &flag;
            ~Unlock() { flag.clear(std::memory_order_release); }
        } unlock{reclaimer};
        if (!external_local) external_local = external_pending.exchange(nullptr, std::memory_order_acquire);
        for (uint32_t i = 0; external_local && i < config.release_batch; ++i) {
            std::unique_ptr<ExternalOwner::Node> node(external_local);
            external_local = node->next;
            node->release(node->data);
            external_bytes.fetch_sub(node->bytes, std::memory_order_relaxed);
            external_released_bytes.fetch_add(node->bytes, std::memory_order_relaxed);
            last_release_thread.store(std::hash<std::thread::id>{}(std::this_thread::get_id()),
                                      std::memory_order_relaxed);
            external_owners.fetch_sub(1, std::memory_order_release);
            // A fence consumer can immediately allocate a replacement. The
            // capacity counter must already be reusable when it sees done.
            node->completion->done.store(true, std::memory_order_release);
        }
        // During normal operation scan a compact bitmap, not every empty/live
        // record in a large CPU arena. Bits are reliable state: a busy reader
        // requeues its bit; no completion queue can overflow or lose a release.
        if (closing.load(std::memory_order_acquire)) {
            const auto scans = std::min(config.scan_slots, config.capacity);
            for (uint32_t i = 0; i < scans; ++i) {
                const auto slot = cursor;
                auto &entry = entries[slot];
                cursor = (cursor + 1 == config.capacity) ? 0 : cursor + 1;
                auto prior = entry.control.load(std::memory_order_acquire);
                if (phase(prior) == Phase::Live &&
                        entry.control.compare_exchange_strong(prior, control(Phase::Retired, refs(prior)),
                            std::memory_order_acq_rel, std::memory_order_acquire)) {
                    // Shutdown retires every owner of a shared payload at once.
                    if (extra_owners(prior)) shared_slots.fetch_sub(1, std::memory_order_relaxed);
                    if (pool_owners(prior)) pool_slots.fetch_sub(1, std::memory_order_relaxed);
                    retired_bytes.fetch_add(config.record_bytes, std::memory_order_relaxed);
                    mark_retired(slot);
                } else if (phase(prior) == Phase::Spilled) {
                    retire_spilled(slot);
                }
            }
        }
        size_t released = 0, examined = 0;
        const auto words = std::min<size_t>(dirty_words, config.scan_slots);
        for (size_t word_count = 0; word_count < words && examined < config.scan_slots &&
                released < config.release_batch; ++word_count) {
            const auto word = dirty_cursor;
            dirty_cursor = (dirty_cursor + 1 == dirty_words) ? 0 : dirty_cursor + 1;
            auto bits = dirty[word].exchange(0, std::memory_order_acq_rel);
            while (bits && examined < config.scan_slots && released < config.release_batch) {
                const auto bit = static_cast<unsigned>(__builtin_ctzll(bits));
                bits &= bits - 1;
                ++examined;
                released += release_one(static_cast<uint32_t>(word * 64 + bit));
            }
            if (bits) dirty[word].fetch_or(bits, std::memory_order_release);
        }
        return released;
    }

    void run() {
        while (true) {
            const bool shutting_down = closing.load(std::memory_order_acquire);
            if (config.background || shutting_down) advance();
            if (shutting_down && used_slots.load(std::memory_order_acquire) == 0 &&
                    writers.load(std::memory_order_acquire) == 0 &&
                    external_owners.load(std::memory_order_acquire) == 0) {
                complete.store(true, std::memory_order_release);
                return;
            }
            std::unique_lock<std::mutex> lock(sleep_mutex);
            wake.wait_for(lock, std::chrono::milliseconds(config.poll_ms));
        }
    }
};

struct HistoricalPayloadArena::Worker {
    explicit Worker(const std::shared_ptr<Shared> &shared)
        : thread([shared] { shared->run(); }) {}
    ~Worker() {
        // Shared, including all payload ownership, remains on the native worker
        // until outstanding readers finish. No Python or device API is used.
        if (thread.joinable()) thread.detach();
    }
    std::thread thread;
};

HistoricalPayloadArena::HistoricalPayloadArena(Config c) {
    if (!c.capacity || !c.record_bytes || !c.poll_ms || !c.scan_slots || !c.release_batch ||
            c.record_bytes > std::numeric_limits<uint64_t>::max() / c.capacity)
        throw std::invalid_argument("Invalid historical CPU payload capacity or worker budget");
    shared_ = std::make_shared<Shared>(c);
    worker_ = std::make_unique<Worker>(shared_);
}

HistoricalPayloadArena::~HistoricalPayloadArena() { close(); }

HistoricalPayloadArena::ExternalOwner::ExternalOwner(std::shared_ptr<Shared> shared,
        std::unique_ptr<Node> node) : shared_(std::move(shared)), node_(std::move(node)) {}

HistoricalPayloadArena::ExternalOwner::~ExternalOwner() {
    auto *node = node_.release();
    if (!node) return;
    if (!admitted_) { delete node; return; }
    auto *head = shared_->external_pending.load(std::memory_order_relaxed);
    do { node->next = head; }
    while (!shared_->external_pending.compare_exchange_weak(head, node,
        std::memory_order_release, std::memory_order_relaxed));
    shared_->wake.notify_one();
}

std::shared_ptr<HistoricalPayloadArena::ExternalCompletion>
HistoricalPayloadArena::ExternalOwner::completion() const { return node_->completion; }

void HistoricalPayloadArena::ExternalOwner::add_bytes(size_t bytes) {
    node_->bytes += bytes;
    shared_->external_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

std::shared_ptr<HistoricalPayloadArena::ExternalOwner>
HistoricalPayloadArena::retain_external(void *data, void (*release)(void *), size_t bytes) {
    if (!data || !release) throw std::invalid_argument("External native storage requires a deleter");
    auto node = std::make_unique<ExternalOwner::Node>();
    node->data = data; node->release = release; node->bytes = bytes;
    // Allocate the owner before admission, so failure never consumes data.
    auto owner = std::shared_ptr<ExternalOwner>(new ExternalOwner(shared_, std::move(node)));
    shared_->writers.fetch_add(1, std::memory_order_acq_rel);
    struct WriterExit {
        Shared &state;
        ~WriterExit() { state.writers.fetch_sub(1, std::memory_order_release); }
    } writer_exit{*shared_};
    auto count = shared_->external_owners.load(std::memory_order_acquire);
    while (true) {
        // Each logical record may own separate transient FP32 sums and FP16
        // means. This bounds descriptors, not bytes: the existing historical
        // admission still reserves the same six index bytes per element.
        // Widen before multiplication; Config::capacity is uint32_t.
        if (shared_->closing.load(std::memory_order_acquire) ||
                count >= uint64_t(shared_->config.capacity) * 2) {
            // Unadmitted owner must not enqueue unowned data.
            owner->node_.reset();
            throw std::runtime_error("External historical storage capacity exhausted or closing");
        }
        if (shared_->external_owners.compare_exchange_weak(count, count + 1,
                std::memory_order_acq_rel, std::memory_order_acquire)) break;
    }
    shared_->external_bytes.fetch_add(bytes, std::memory_order_relaxed);
    owner->admitted_ = true;
    return owner;
}

HistoricalPayloadArena::Read::Read(std::shared_ptr<Shared> shared, Handle handle)
    : shared_(std::move(shared)), handle_(handle) {}
HistoricalPayloadArena::Read::~Read() { reset(); }
HistoricalPayloadArena::Read::Read(Read &&other) noexcept
    : shared_(std::move(other.shared_)), handle_(other.handle_) { other.handle_ = {}; }
HistoricalPayloadArena::Read &HistoricalPayloadArena::Read::operator=(Read &&other) noexcept {
    if (this != &other) {
        reset();
        shared_ = std::move(other.shared_);
        handle_ = other.handle_;
        other.handle_ = {};
    }
    return *this;
}
void HistoricalPayloadArena::Read::reset() noexcept {
    if (!shared_) return;
    shared_->readers.fetch_sub(1, std::memory_order_relaxed);
    shared_->entries[handle_.slot].control.fetch_sub(1, std::memory_order_release);
    shared_.reset();
    handle_ = {};
}
const uint8_t *HistoricalPayloadArena::Read::data() const {
    if (!shared_) throw std::logic_error("Empty historical CPU payload read");
    return shared_->entries[handle_.slot].payload.get();
}
size_t HistoricalPayloadArena::Read::size() const {
    if (!shared_) throw std::logic_error("Empty historical CPU payload read");
    return shared_->config.record_bytes;
}
HistoricalPayloadArena::Read HistoricalPayloadArena::Read::clone() const {
    if (!shared_) return {};
    auto &entry = shared_->entries[handle_.slot];
    auto prior = entry.control.load(std::memory_order_acquire);
    for (unsigned attempt = 0; attempt < 16; ++attempt) {
        if (!refs(prior) || (phase(prior) != Phase::Live && phase(prior) != Phase::Retired))
            throw std::logic_error("Historical CPU operation lost its read ownership");
        if (refs(prior) == refs_mask) throw std::overflow_error("Historical CPU reader overflow");
        if (entry.control.compare_exchange_weak(prior, prior + 1,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            shared_->readers.fetch_add(1, std::memory_order_relaxed);
            return Read(shared_, handle_);
        }
    }
    throw std::runtime_error("Historical CPU read clone must retry after contention");
}

void HistoricalPayloadArena::publish(Handle h, const void *source, size_t bytes) {
    const Segment segment{source, bytes};
    publish_segments(h, &segment, 1);
}

void HistoricalPayloadArena::publish_segments(Handle h, const Segment *segments, size_t count) {
    auto shared = shared_;
    shared->writers.fetch_add(1, std::memory_order_acq_rel);
    struct WriterExit {
        Shared &shared;
        ~WriterExit() { shared.writers.fetch_sub(1, std::memory_order_release); }
    } writer{*shared};
    if (shared->closing.load(std::memory_order_acquire))
        throw std::runtime_error("Historical CPU payload arena is closing");
    size_t bytes = 0;
    if (!segments || !count)
        throw std::invalid_argument("Historical CPU publication requires source segments");
    for (size_t i = 0; i < count; ++i) {
        if ((segments[i].size && !segments[i].data) ||
                segments[i].size > shared->config.record_bytes - bytes)
            throw std::invalid_argument("Invalid historical CPU payload segment extent");
        bytes += segments[i].size;
    }
    if (h.slot >= shared->config.capacity || !h.generation || bytes != shared->config.record_bytes)
        throw std::invalid_argument("Invalid historical CPU payload publication");
    auto &entry = shared->entries[h.slot];
    auto expected = control(Phase::Empty);
    if (!entry.control.compare_exchange_strong(expected, control(Phase::Initializing),
            std::memory_order_acq_rel, std::memory_order_acquire))
        throw std::runtime_error("Historical CPU payload slot is not reusable");
    try {
        if (h.generation <= entry.generation.load(std::memory_order_relaxed))
            throw std::invalid_argument("Historical CPU payload generation must increase");
        Payload payload = shared->pool ? Payload(shared->pool->acquire(), PayloadFree{shared->pool.get()})
                                       : Payload(std::make_unique<uint8_t[]>(bytes).release(), PayloadFree{});
        size_t offset = 0;
        for (size_t i = 0; i < count; ++i) {
            if (segments[i].size) std::memcpy(payload.get() + offset, segments[i].data, segments[i].size);
            offset += segments[i].size;
        }
        entry.payload = std::move(payload);
        entry.generation.store(h.generation, std::memory_order_relaxed);
        shared->used_slots.fetch_add(1, std::memory_order_relaxed);
        shared->used_bytes.fetch_add(bytes, std::memory_order_relaxed);
        const auto next = shared->closing.load(std::memory_order_acquire) ? Phase::Retired : Phase::Live;
        if (next == Phase::Retired) shared->retired_bytes.fetch_add(bytes, std::memory_order_relaxed);
        entry.control.store(control(next), std::memory_order_release);
        if (next == Phase::Retired) shared->mark_retired(h.slot);
    } catch (...) {
        entry.control.store(control(Phase::Empty), std::memory_order_release);
        throw;
    }
}

HistoricalPayloadArena::Read HistoricalPayloadArena::acquire(Handle h) const {
    auto shared = shared_;
    if (h.slot >= shared->config.capacity || !h.generation ||
            shared->closing.load(std::memory_order_acquire)) return {};
    auto &entry = shared->entries[h.slot];
    auto prior = entry.control.load(std::memory_order_acquire);
    for (unsigned attempt = 0; attempt < 16; ++attempt) {
        if (phase(prior) != Phase::Live || refs(prior) == refs_mask) return {};
        if (!entry.control.compare_exchange_weak(prior, prior + 1,
                std::memory_order_acq_rel, std::memory_order_acquire)) continue;
        if (entry.generation.load(std::memory_order_relaxed) != h.generation ||
                shared->closing.load(std::memory_order_acquire)) {
            entry.control.fetch_sub(1, std::memory_order_release);
            return {};
        }
        shared->readers.fetch_add(1, std::memory_order_relaxed);
        return Read(std::move(shared), h);
    }
    return {};
}

HistoricalPayloadArena::RetireResult HistoricalPayloadArena::retire(Handle h) {
    return shared_->drop(h, DropKind::Plain);
}

HistoricalPayloadArena::RetireResult HistoricalPayloadArena::release_pool(Handle h) {
    return shared_->drop(h, DropKind::Pool);
}

HistoricalPayloadArena::RetireResult HistoricalPayloadArena::retire_unshared(Handle h) {
    return shared_->drop(h, DropKind::Unshared);
}

// One owner leaves (Plain: a publisher or alias descriptor, Pool: a shared-prefix
// pool owner), or, for Unshared, the historical GC retires a payload only if its
// sole owner is plain. The last owner's retirement, and every single-owner path,
// is the original retirement unchanged.
HistoricalPayloadArena::RetireResult HistoricalPayloadArena::Shared::drop(Handle h, DropKind kind) {
    auto &shared = *this;
    if (h.slot >= shared.config.capacity || !h.generation) return RetireResult::StaleHandle;
    auto &entry = shared.entries[h.slot];
    auto prior = entry.control.load(std::memory_order_acquire);
    if (phase(prior) == Phase::Spilled) {
        // Spilled payloads always have a single plain owner (spill refuses others).
        if (entry.generation.load(std::memory_order_relaxed) != h.generation || kind == DropKind::Pool)
            return RetireResult::StaleHandle;
        return shared.retire_spilled(h.slot) ? RetireResult::Committed : RetireResult::Retry;
    }
    for (unsigned attempt = 0; attempt < 16; ++attempt) {
        if (phase(prior) != Phase::Live && phase(prior) != Phase::Retired)
            return RetireResult::StaleHandle;
        if (refs(prior) == refs_mask) return RetireResult::Retry;
        if (!entry.control.compare_exchange_weak(prior, prior + 1,
                std::memory_order_acq_rel, std::memory_order_acquire)) continue;
        struct Guard {
            std::atomic<uint64_t> &value;
            ~Guard() { value.fetch_sub(1, std::memory_order_release); }
        } guard{entry.control};
        if (entry.generation.load(std::memory_order_relaxed) != h.generation)
            return RetireResult::StaleHandle;
        prior = entry.control.load(std::memory_order_acquire);
        for (unsigned retry = 0; retry < 16; ++retry) {
            if (phase(prior) == Phase::Retired) return RetireResult::AlreadyRetired;
            const auto extra = extra_owners(prior), pools = pool_owners(prior);
            if (kind == DropKind::Unshared) {
                if (extra || pools) return RetireResult::Shared;   // other owners: change nothing
            } else if (kind == DropKind::Pool ? !pools : extra + 1 == pools) {
                return RetireResult::StaleHandle;   // no owner of this kind is left to drop
            }
            if (extra) {
                const auto next = prior - owner_unit - (kind == DropKind::Pool ? pool_unit : 0);
                if (entry.control.compare_exchange_weak(prior, next,
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                    if (extra == 1) shared.shared_slots.fetch_sub(1, std::memory_order_relaxed);
                    if (kind == DropKind::Pool && pools == 1)
                        shared.pool_slots.fetch_sub(1, std::memory_order_relaxed);
                    return RetireResult::Shared;
                }
                continue;
            }
            if (entry.control.compare_exchange_weak(prior, control(Phase::Retired, refs(prior)),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                if (pools) shared.pool_slots.fetch_sub(1, std::memory_order_relaxed);
                shared.retired_bytes.fetch_add(shared.config.record_bytes, std::memory_order_relaxed);
                shared.mark_retired(h.slot);
                return RetireResult::Committed;
            }
        }
        return RetireResult::Retry;
    }
    return RetireResult::Retry;
}

bool HistoricalPayloadArena::share(Handle h, bool pool) {
    auto shared = shared_;
    if (h.slot >= shared->config.capacity || !h.generation ||
            shared->closing.load(std::memory_order_acquire)) return false;
    auto &entry = shared->entries[h.slot];
    auto prior = entry.control.load(std::memory_order_acquire);
    while (true) {
        if (phase(prior) != Phase::Live || refs(prior) == refs_mask) return false;
        // A temporary reader keeps the slot's generation stable while it is checked.
        if (!entry.control.compare_exchange_weak(prior, prior + 1,
                std::memory_order_acq_rel, std::memory_order_acquire)) continue;
        break;
    }
    struct Guard {
        std::atomic<uint64_t> &value;
        ~Guard() { value.fetch_sub(1, std::memory_order_release); }
    } guard{entry.control};
    if (entry.generation.load(std::memory_order_relaxed) != h.generation ||
            shared->closing.load(std::memory_order_acquire)) return false;
    prior = entry.control.load(std::memory_order_acquire);
    while (true) {
        if (phase(prior) != Phase::Live) return false;
        const auto extra = extra_owners(prior), pools = pool_owners(prior);
        if (extra == owner_max || (pool && pools == pool_max)) return false;
        if (entry.control.compare_exchange_weak(prior, prior + owner_unit + (pool ? pool_unit : 0),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            if (!extra) shared->shared_slots.fetch_add(1, std::memory_order_relaxed);
            if (pool && !pools) shared->pool_slots.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
}

HistoricalPayloadArena::Owners HistoricalPayloadArena::owners(Handle h) const {
    Owners result;
    if (h.slot >= shared_->config.capacity || !h.generation) return result;
    const auto &entry = shared_->entries[h.slot];
    const auto value = entry.control.load(std::memory_order_acquire);
    if (entry.generation.load(std::memory_order_relaxed) != h.generation) return result;
    if (phase(value) == Phase::Live || phase(value) == Phase::Spilled) {
        result.total = static_cast<uint32_t>(extra_owners(value) + 1);
        result.pool = static_cast<uint32_t>(pool_owners(value));
    }
    return result;
}

bool HistoricalPayloadArena::spill(Handle h) {
    auto shared = shared_;
    if (h.slot >= shared->config.capacity || !h.generation ||
            shared->closing.load(std::memory_order_acquire)) return false;
    auto &entry = shared->entries[h.slot];
    auto expected = control(Phase::Live);               // no readers
    if (!entry.control.compare_exchange_strong(expected, control(Phase::Initializing),
            std::memory_order_acq_rel, std::memory_order_acquire)) return false;
    if (entry.generation.load(std::memory_order_relaxed) != h.generation) {
        entry.control.store(control(Phase::Live), std::memory_order_release);
        return false;
    }
    entry.payload.reset();
    shared->used_bytes.fetch_sub(shared->config.record_bytes, std::memory_order_relaxed);
    shared->spilled_slots.fetch_add(1, std::memory_order_relaxed);
    shared->spilled_bytes.fetch_add(shared->config.record_bytes, std::memory_order_relaxed);
    entry.control.store(control(Phase::Spilled), std::memory_order_release);
    return true;
}

bool HistoricalPayloadArena::restore(Handle h, const Segment *segments, size_t count) {
    auto shared = shared_;
    size_t bytes = 0;
    if (!segments || !count)
        throw std::invalid_argument("Historical CPU restore requires source segments");
    for (size_t i = 0; i < count; ++i) {
        if ((segments[i].size && !segments[i].data) ||
                segments[i].size > shared->config.record_bytes - bytes)
            throw std::invalid_argument("Invalid historical CPU restore segment extent");
        bytes += segments[i].size;
    }
    if (h.slot >= shared->config.capacity || !h.generation || bytes != shared->config.record_bytes)
        throw std::invalid_argument("Invalid historical CPU payload restore");
    auto &entry = shared->entries[h.slot];
    auto expected = control(Phase::Spilled);
    if (!entry.control.compare_exchange_strong(expected, control(Phase::Initializing),
            std::memory_order_acq_rel, std::memory_order_acquire)) return false;
    if (entry.generation.load(std::memory_order_relaxed) != h.generation) {
        entry.control.store(control(Phase::Spilled), std::memory_order_release);
        return false;
    }
    try {
        Payload payload = shared->pool ? Payload(shared->pool->acquire(), PayloadFree{shared->pool.get()})
                                       : Payload(std::make_unique<uint8_t[]>(bytes).release(), PayloadFree{});
        size_t offset = 0;
        for (size_t i = 0; i < count; ++i) {
            if (segments[i].size) std::memcpy(payload.get() + offset, segments[i].data, segments[i].size);
            offset += segments[i].size;
        }
        entry.payload = std::move(payload);
    } catch (...) {
        entry.control.store(control(Phase::Spilled), std::memory_order_release);
        throw;
    }
    shared->used_bytes.fetch_add(bytes, std::memory_order_relaxed);
    shared->spilled_slots.fetch_sub(1, std::memory_order_relaxed);
    shared->spilled_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    entry.control.store(control(Phase::Live), std::memory_order_release);
    return true;
}

bool HistoricalPayloadArena::spilled(Handle h) const {
    if (h.slot >= shared_->config.capacity || !h.generation) return false;
    auto &entry = shared_->entries[h.slot];
    return phase(entry.control.load(std::memory_order_acquire)) == Phase::Spilled &&
        entry.generation.load(std::memory_order_relaxed) == h.generation;
}

bool HistoricalPayloadArena::reclaimed(Handle h) const {
    return h.slot < shared_->config.capacity && h.generation &&
        shared_->entries[h.slot].released_generation.load(std::memory_order_acquire) >= h.generation;
}
bool HistoricalPayloadArena::owns(const Read &read) const {
    return bool(read) && read.shared_ == shared_;
}
uint32_t HistoricalPayloadArena::capacity() const { return shared_->config.capacity; }
size_t HistoricalPayloadArena::record_bytes() const { return shared_->config.record_bytes; }
HistoricalPayloadArena::Stats HistoricalPayloadArena::stats() const {
    Stats result;
#define LOAD(name) result.name = shared_->name.load(std::memory_order_relaxed)
    LOAD(used_slots); LOAD(used_bytes); LOAD(retired_bytes); LOAD(readers);
    LOAD(released_slots); LOAD(released_bytes); LOAD(last_release_thread);
    LOAD(external_owners); LOAD(external_bytes); LOAD(external_released_bytes);
    LOAD(spilled_slots); LOAD(spilled_bytes); LOAD(shared_slots); LOAD(pool_slots);
#undef LOAD
    result.unique_bytes = result.used_bytes;   // every resident payload is counted once
    if (shared_->pool) {
        result.pool_hits = shared_->pool->hits();
        result.pool_misses = shared_->pool->misses();
        result.pool_reserved_bytes = shared_->pool->reserved();
    }
    return result;
}
size_t HistoricalPayloadArena::advance() { return shared_->advance(); }
bool HistoricalPayloadArena::close() {
    shared_->closing.store(true, std::memory_order_release);
    shared_->wake.notify_one();
    return shared_->complete.load(std::memory_order_acquire);
}

} // namespace vllm_kvmem
