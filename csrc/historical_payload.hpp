// CPU payload lifetime, independent of GPU execution and repository metadata.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <atomic>

namespace vllm_kvmem {

class HistoricalPayloadArena {
    struct Shared;
public:
    struct Handle {
        uint32_t slot = UINT32_MAX;
        uint64_t generation = 0;
    };
    struct Segment { const void *data; size_t size; };
    struct Config {
        uint32_t capacity = 0;
        size_t record_bytes = 0;
        uint32_t poll_ms = 10, scan_slots = 256, release_batch = 64;
        // Manual advance is for deterministic tests. Closing still activates a
        // native reaper so a final foreground Read never destroys payloads.
        bool background = true;
        // Opt-in record pool (0 = per-record heap allocation, the original
        // behavior). Records are carved from huge-page-advised slabs of this
        // many MiB; released records are reused LIFO (already faulted), and a
        // native filler pre-faults `pool_ahead_records` free records so a
        // publication never pays first-touch page faults. Bytes are identical.
        uint32_t pool_chunk_mib = 0;
        uint32_t pool_ahead_records = 0;
    };
    // Shared: a non-last owner dropped its reference; the payload stays Live
    // for its other owners (shared prefixes, docs/prefix-sharing.md phase 2).
    enum class RetireResult { Committed, AlreadyRetired, StaleHandle, Retry, Shared };
    struct Stats {
        uint64_t used_slots = 0, used_bytes = 0, retired_bytes = 0;
        uint64_t readers = 0, released_slots = 0, released_bytes = 0;
        uint64_t last_release_thread = 0;
        uint64_t external_owners = 0, external_bytes = 0, external_released_bytes = 0;
        uint64_t pool_hits = 0, pool_misses = 0, pool_reserved_bytes = 0;
        uint64_t spilled_slots = 0, spilled_bytes = 0;
        // Shared prefixes: payloads with two or more owners, payloads held by a
        // shared-prefix pool owner, and resident payload bytes counted once.
        uint64_t shared_slots = 0, pool_slots = 0, unique_bytes = 0;
    };
    struct Owners { uint32_t total = 0, pool = 0; };
    // C-only storage deleters (e.g. native index arrays). The last
    // foreground owner enqueues; only the arena worker invokes the deleter.
    struct ExternalCompletion {
        std::atomic<bool> done{false};
    };
    class ExternalOwner {
    public:
        ~ExternalOwner();
        ExternalOwner(const ExternalOwner &) = delete;
        std::shared_ptr<ExternalCompletion> completion() const;
        void add_bytes(size_t);
    private:
        friend class HistoricalPayloadArena;
        struct Node;
        ExternalOwner(std::shared_ptr<Shared>, std::unique_ptr<Node>);
        std::shared_ptr<Shared> shared_;
        std::unique_ptr<Node> node_;
        bool admitted_ = false;
    };
    class Read {
    public:
        Read() = default;
        ~Read();
        Read(Read &&) noexcept;
        Read &operator=(Read &&) noexcept;
        Read(const Read &) = delete;
        Read &operator=(const Read &) = delete;
        explicit operator bool() const { return bool(shared_); }
        const uint8_t *data() const;
        size_t size() const;
        Handle handle() const { return handle_; }
        // Clone rights for this already admitted operation, not a new read.
        Read clone() const;
        void reset() noexcept;
    private:
        friend class HistoricalPayloadArena;
        Read(std::shared_ptr<Shared>, Handle);
        std::shared_ptr<Shared> shared_;
        Handle handle_;
    };

    explicit HistoricalPayloadArena(Config);
    ~HistoricalPayloadArena();
    HistoricalPayloadArena(const HistoricalPayloadArena &) = delete;
    HistoricalPayloadArena &operator=(const HistoricalPayloadArena &) = delete;
    // Called by the existing archive producer, never by the model thread.
    // Owns an independent immutable byte copy after return.
    void publish(Handle, const void *, size_t);
    void publish_segments(Handle, const Segment *, size_t);
    Read acquire(Handle) const;
    bool owns(const Read &) const;
    // Shared prefixes (docs/prefix-sharing.md, phase 2). A published payload has
    // one owner, its publisher. share() adds an owner to a Live payload; pool
    // marks a shared-prefix pool owner (an image). retire() drops one plain
    // owner: Shared while others remain, the original retirement for the last.
    // release_pool() drops one pool owner the same way. retire_unshared() is the
    // historical GC's retirement: it retires only a payload with one plain owner
    // and otherwise reports Shared without changing anything. A single-owner
    // payload behaves exactly as before sharing existed.
    bool share(Handle, bool pool = false);
    RetireResult release_pool(Handle);
    RetireResult retire_unshared(Handle);
    Owners owners(Handle) const;
    std::shared_ptr<ExternalOwner> retain_external(void *, void (*)(void *), size_t bytes);
    uint32_t capacity() const;
    size_t record_bytes() const;
    RetireResult retire(Handle);
    bool reclaimed(Handle) const;
    // Opt-in session tiering (CPU TTL to SSD). The caller keeps an external
    // copy of the bytes. spill: a Live record with no readers drops its CPU
    // bytes but keeps its handle (false when busy, stale or closing).
    // restore: the identical bytes come back under the same handle.
    bool spill(Handle);
    bool restore(Handle, const Segment *, size_t);
    bool spilled(Handle) const;
    Stats stats() const;
    // Sole native reclaimer (or deterministic test driver); no Python callback.
    size_t advance();
    // Nonblocking shutdown. New reads/writes close immediately; existing reads
    // finish safely. Returns true only after native cleanup has completed.
    bool close();
private:
    struct Worker;
    std::shared_ptr<Shared> shared_;
    std::unique_ptr<Worker> worker_;
};

} // namespace vllm_kvmem
