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

namespace vllm_kvmem {
namespace {
constexpr unsigned shift = 60;
constexpr uint64_t refs_mask = (uint64_t{1} << shift) - 1;
enum class Phase : uint64_t { Empty, Initializing, Live, Retired, Releasing };
constexpr uint64_t control(Phase p, uint64_t refs = 0) { return (uint64_t(p) << shift) | refs; }
Phase phase(uint64_t value) { return Phase(value >> shift); }
uint64_t refs(uint64_t value) { return value & refs_mask; }
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "Historical payload lifetime requires lock-free 64-bit atomics");
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
        std::unique_ptr<uint8_t[]> payload;
    };
    explicit Shared(Config c)
        : config(c), entries(new Entry[c.capacity]), dirty_words((uint64_t(c.capacity) + 63) / 64),
          dirty(new std::atomic<uint64_t>[dirty_words]) {
        for (size_t i = 0; i < dirty_words; ++i) dirty[i].store(0, std::memory_order_relaxed);
    }
    Config config;
    std::unique_ptr<Entry[]> entries;
    size_t dirty_words;
    std::unique_ptr<std::atomic<uint64_t>[]> dirty;
    std::atomic<bool> closing{false}, complete{false};
    std::atomic_flag reclaimer = ATOMIC_FLAG_INIT;
    std::atomic<uint64_t> used_slots{0}, used_bytes{0}, retired_bytes{0}, readers{0};
    std::atomic<uint64_t> released_slots{0}, released_bytes{0}, last_release_thread{0}, writers{0};
    std::atomic<uint64_t> external_owners{0}, external_bytes{0}, external_released_bytes{0};
    std::atomic<ExternalOwner::Node *> external_pending{nullptr};
    ExternalOwner::Node *external_local = nullptr; // Sole reclaimer ownership.
    uint32_t cursor = 0; // Exclusive reclaimer ownership, not a foreground lock.
    size_t dirty_cursor = 0;
    std::mutex sleep_mutex;
    std::condition_variable wake;

    void mark_retired(uint32_t slot) {
        dirty[slot / 64].fetch_or(uint64_t{1} << (slot % 64), std::memory_order_release);
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
                    retired_bytes.fetch_add(config.record_bytes, std::memory_order_relaxed);
                    mark_retired(slot);
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
        auto payload = std::make_unique<uint8_t[]>(bytes);
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
    auto shared = shared_;
    if (h.slot >= shared->config.capacity || !h.generation) return RetireResult::StaleHandle;
    auto &entry = shared->entries[h.slot];
    auto prior = entry.control.load(std::memory_order_acquire);
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
            if (entry.control.compare_exchange_weak(prior, control(Phase::Retired, refs(prior)),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                shared->retired_bytes.fetch_add(shared->config.record_bytes, std::memory_order_relaxed);
                shared->mark_retired(h.slot);
                return RetireResult::Committed;
            }
        }
        return RetireResult::Retry;
    }
    return RetireResult::Retry;
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
#undef LOAD
    return result;
}
size_t HistoricalPayloadArena::advance() { return shared_->advance(); }
bool HistoricalPayloadArena::close() {
    shared_->closing.store(true, std::memory_order_release);
    shared_->wake.notify_one();
    return shared_->complete.load(std::memory_order_acquire);
}

} // namespace vllm_kvmem
