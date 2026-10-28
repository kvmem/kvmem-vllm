#include "historical_manager.hpp"
#include "historical_redundancy.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>

namespace vllm_kvmem {
namespace {
using Repo = qw3::KvMemRepository;
using Handle = qw3::KvMemBlockHandle;
using Payload = HistoricalPayloadArena;

#define HISTORY_STATS(X) \
    X(published) X(ticks) X(examined) X(decisions) X(retired) X(cpu_released) \
    X(slots_released) X(pending_slots) X(pending_cpu_bytes) X(retries) X(stale) \
    X(policy_failures) X(shadow_decisions) X(admission_attempts) X(admission_retries) \
    X(redundancy_retired) X(lossy_retired) X(rejected_ineligible) X(cancelled_decisions)

struct AtomicStats {
#define FIELD(name) std::atomic<uint64_t> name{0};
    HISTORY_STATS(FIELD)
#undef FIELD
    HistoricalGcStats snapshot() const {
        HistoricalGcStats result;
#define LOAD(name) result.name = name.load(std::memory_order_relaxed);
        HISTORY_STATS(LOAD)
#undef LOAD
        return result;
    }
};

struct PublishedBlock {
    // Every concurrently sampled field is atomic. The version validates a
    // coherent row; it does not disguise non-atomic concurrent accesses.
    std::atomic<uint64_t> version{0}, generation{0}, logical_id{0}, birth{0}, last_use{0};
    std::atomic<uint64_t> payload_generation{0};
    std::atomic<uint32_t> payload_slot{UINT32_MAX}, flags{0};
    // Engine metadata has its own sole-producer channel. An I/O producer may
    // publish backing at the same time without making decode acquire its guard.
    std::atomic<uint64_t> engine_version{0}, engine_generation{0}, engine_logical_id{0}, engine_use{0};
    std::atomic<uint32_t> engine_flags{0};
    std::atomic<uint32_t> policy_holds{0};
    std::atomic_flag policy_gate = ATOMIC_FLAG_INIT;
    std::atomic<uint64_t> cpu_event{0}, cpu_ack{0}, slot_event{0}, slot_ack{0};
};

struct BlockView {
    HistoricalBlockMeta meta;
    Payload::Handle payload;
};

struct PendingBlock {
    Handle handle;
    Payload::Handle payload;
    bool active = false, cpu_done = false, slot_done = false, byte_credit = false;
};

bool sample(const PublishedBlock &source, uint32_t slot, BlockView &out) {
    const auto version = source.version.load(std::memory_order_acquire);
    if (version & 1) return false;
    out.meta.handle = {slot, source.generation.load(std::memory_order_relaxed)};
    out.meta.logical_id = source.logical_id.load(std::memory_order_relaxed);
    out.meta.birth_seq = source.birth.load(std::memory_order_relaxed);
    out.meta.last_use_seq = source.last_use.load(std::memory_order_relaxed);
    auto flags = source.flags.load(std::memory_order_relaxed);
    const auto engine_version = source.engine_version.load(std::memory_order_acquire);
    if (engine_version & 1) return false;
    const auto engine_generation = source.engine_generation.load(std::memory_order_relaxed);
    if (engine_generation && engine_generation >= out.meta.handle.generation) {
        const bool same = engine_generation == out.meta.handle.generation;
        out.meta.handle.generation = engine_generation;
        out.meta.logical_id = source.engine_logical_id.load(std::memory_order_relaxed);
        out.meta.birth_seq = out.meta.logical_id;
        out.meta.last_use_seq = same ? std::max(out.meta.last_use_seq, source.engine_use.load(std::memory_order_relaxed))
                                    : source.engine_use.load(std::memory_order_relaxed);
        flags = source.engine_flags.load(std::memory_order_relaxed);
    }
    out.meta.sealed = flags & HistoricalSealed;
    out.meta.mandatory = flags & HistoricalMandatory;
    out.meta.gpu_resident = flags & HistoricalGpu;
    out.meta.mandatory |= source.policy_holds.load(std::memory_order_acquire) != 0;
    out.payload = {source.payload_slot.load(std::memory_order_relaxed),
                   source.payload_generation.load(std::memory_order_relaxed)};
    if (out.meta.handle.generation != source.generation.load(std::memory_order_relaxed)) out.payload = {};
    // LRU's sealed history is a finalized immutable archive record. Engine
    // sealing alone may precede a queued I/O publisher that still owns bytes.
    out.meta.sealed = out.meta.sealed && out.payload.generation != 0;
    return source.version.load(std::memory_order_acquire) == version &&
           source.engine_version.load(std::memory_order_acquire) == engine_version;
}

bool begin_publish(PublishedBlock &block, uint64_t &version) {
    version = block.version.load(std::memory_order_acquire);
    return !(version & 1) && block.version.compare_exchange_strong(version, version + 1,
            std::memory_order_acq_rel, std::memory_order_acquire);
}
}

struct HistoricalScopeStorage {
    HistoricalScopeStorage(size_t capacity, const HistoricalGcWorker::Config &config)
        : capacity(capacity), records(new PublishedBlock[capacity]), jobs(capacity),
          completion_words((capacity+63)/64), completion_bits(new std::atomic<uint64_t>[completion_words]),
          shadow(capacity, 0),
          features(make_historical_feature_source(config.policy, config.policy_params, capacity)),
          policy(make_historical_kv_policy(config.policy, config.policy_params, capacity, config.keep_mandatory)),
          batch(config.delete_blocks) {
        snapshot.blocks.reserve(capacity);
        for (size_t i = 0; i < completion_words; ++i) completion_bits[i].store(0, std::memory_order_relaxed);
    }
    void reset() {
        for (size_t i = 0; i < capacity; ++i) {
            records[i].~PublishedBlock();
            new (&records[i]) PublishedBlock();
            jobs[i] = {};
            shadow[i] = 0;
        }
        for (size_t i = 0; i < completion_words; ++i) completion_bits[i].store(0, std::memory_order_relaxed);
        policy->reset();
        if (features) features->reset();
        batch.clear_after_ack();
        snapshot.blocks.clear();
        snapshot.incarnation = snapshot.round = 0;
    }
    size_t capacity;
    std::unique_ptr<PublishedBlock[]> records;
    std::vector<PendingBlock> jobs;
    size_t completion_words;
    std::unique_ptr<std::atomic<uint64_t>[]> completion_bits;
    std::vector<uint64_t> shadow;
    std::shared_ptr<HistoricalFeatureSource> features;
    std::unique_ptr<HistoricalKvPolicy> policy;
    HistoricalDecisionBatch batch;
    HistoricalSnapshot snapshot;
};

struct HistoricalScopeState {
    HistoricalScopeState(std::shared_ptr<Repo> r, std::shared_ptr<Payload> a,
                         HistoricalGcWorker::Config c, uint64_t epoch,
                         std::unique_ptr<HistoricalKvPolicy> p, std::shared_ptr<HistoricalScopeStorage> buffers)
        : repository(std::move(r)), arena(std::move(a)), config(c), incarnation(epoch),
          storage(std::move(buffers)), records(storage->records.get()), jobs(storage->jobs),
          completion_words(storage->completion_words), completion_bits(storage->completion_bits.get()),
          shadow(storage->shadow), features(storage->features), custom_policy(std::move(p)),
          policy(custom_policy ? custom_policy.get() : storage->policy.get()), batch(storage->batch), snapshot(storage->snapshot) {
        if (config.ready_high > repository->capacity())
            throw std::invalid_argument("Historical scope watermarks exceed repository capacity");
        snapshot.incarnation = incarnation;
        policy->bind_features(features);
    }
    std::shared_ptr<Repo> repository;
    std::shared_ptr<Payload> arena;
    const HistoricalGcWorker::Config config;
    const uint64_t incarnation;
    std::shared_ptr<HistoricalScopeStorage> storage;
    PublishedBlock *records;
    std::vector<PendingBlock> &jobs;
    const size_t completion_words;
    std::atomic<uint64_t> *completion_bits;
    std::vector<uint64_t> &shadow;
    std::shared_ptr<HistoricalFeatureSource> features;
    std::unique_ptr<HistoricalKvPolicy> custom_policy;
    HistoricalKvPolicy *policy;
    HistoricalDecisionBatch &batch;
    HistoricalSnapshot &snapshot;
    AtomicStats stats;
    std::atomic<uint64_t> use_stamp{0};
    std::atomic<uint64_t> history_end{0}, recent_begin{0};
    std::atomic<uint32_t> requested_ready{0};
    std::atomic<uint64_t> requested_cpu_limit{UINT64_MAX};
    std::atomic<bool> closing{false}, resources_closed{false}, paused{false};
    std::atomic<uint64_t> pause_request{0}, pause_ack{UINT64_MAX};
    size_t snapshot_cursor = 0, release_cursor = 0, close_cursor = 0, batch_cursor = 0;
    bool round_active = false, snapshot_ready = false, policy_round_done = false;
    uint32_t shadow_slots = 0;
    uint64_t shadow_bytes = 0;

    void signal_completion(uint32_t slot) {
        completion_bits[slot / 64].fetch_or(uint64_t{1} << (slot % 64), std::memory_order_release);
    }

    void clear_completion_if_acknowledged(uint32_t slot) {
        auto &entry = records[slot];
        completion_bits[slot / 64].fetch_and(~(uint64_t{1} << (slot % 64)), std::memory_order_acq_rel);
        // A backend publication may race the clear. Its release RMW either
        // follows this clear, or is acquired above and observed by this check.
        if (entry.cpu_event.load(std::memory_order_acquire) > entry.cpu_ack.load(std::memory_order_acquire) ||
                entry.slot_event.load(std::memory_order_acquire) > entry.slot_ack.load(std::memory_order_acquire))
            signal_completion(slot);
    }

    bool publish(const Repo::Lease &lease, const Payload::Read *cpu, uint32_t flags, uint64_t use) {
        if (!repository->owns(lease)) throw std::invalid_argument("Historical publication has a foreign repository lease");
        if (cpu && !arena->owns(*cpu)) throw std::invalid_argument("Historical publication has foreign CPU backing");
        const auto &record = lease.record();
        auto &destination = records[record.handle.slot];
        if (!cpu) {
            auto version = destination.engine_version.load(std::memory_order_acquire);
            if ((version & 1) || !destination.engine_version.compare_exchange_strong(version, version + 1,
                    std::memory_order_acq_rel, std::memory_order_acquire)) return false;
            const bool same = destination.engine_generation.load(std::memory_order_relaxed) == record.handle.generation;
            const auto old_flags = same ? destination.engine_flags.load(std::memory_order_relaxed) : 0;
            auto last = same ? destination.engine_use.load(std::memory_order_relaxed) : 0;
            if ((old_flags & HistoricalGpu) || (flags & HistoricalGpu))
                last = std::max(last, use_stamp.load(std::memory_order_acquire));
            destination.engine_logical_id.store(record.logical_id, std::memory_order_relaxed);
            destination.engine_use.store(std::max(last, use), std::memory_order_relaxed);
            destination.engine_flags.store(flags, std::memory_order_relaxed);
            destination.engine_generation.store(record.handle.generation, std::memory_order_relaxed);
            destination.engine_version.store(version + 2, std::memory_order_release);
            return true;
        }
        uint64_t version;
        if (!begin_publish(destination, version)) return false;
        struct PublishExit {
            PublishedBlock &block;
            uint64_t version;
            ~PublishExit() { block.version.store(version + 2, std::memory_order_release); }
        } unlock{destination, version};
        const bool same = destination.generation.load(std::memory_order_relaxed) == record.handle.generation;
        const auto previous_payload = same ? destination.payload_generation.load(std::memory_order_relaxed) : 0;
        if (cpu && previous_payload &&
                (previous_payload != cpu->handle().generation ||
                 destination.payload_slot.load(std::memory_order_relaxed) != cpu->handle().slot))
            throw std::invalid_argument("Cannot overwrite immutable historical backing");
        const auto old_flags = same ? destination.flags.load(std::memory_order_relaxed) : 0;
        auto last = same ? destination.last_use.load(std::memory_order_relaxed) : 0;
        if ((old_flags & HistoricalGpu) && !(flags & HistoricalGpu) && !cpu)
            last = std::max(last, use_stamp.load(std::memory_order_acquire));
        if (!same) {
            destination.payload_slot.store(UINT32_MAX, std::memory_order_relaxed);
            destination.payload_generation.store(0, std::memory_order_relaxed);
            destination.birth.store(record.logical_id, std::memory_order_relaxed);
        }
        if (cpu) {
            const auto handle = cpu->handle();
            destination.payload_slot.store(handle.slot, std::memory_order_relaxed);
            destination.payload_generation.store(handle.generation, std::memory_order_relaxed);
            // Backing may complete on the I/O producer after a window update.
            // Preserve the engine's existing residency observation.
            if (same) flags = (flags & ~HistoricalGpu) | (old_flags & HistoricalGpu);
        }
        destination.generation.store(record.handle.generation, std::memory_order_relaxed);
        destination.logical_id.store(record.logical_id, std::memory_order_relaxed);
        destination.last_use.store(std::max(last, use), std::memory_order_relaxed);
        destination.flags.store(flags, std::memory_order_relaxed);
        if (cpu && !previous_payload) stats.published.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void credit_cpu(PendingBlock &job, uint64_t &global_pending) {
        if (job.payload.generation && !job.byte_credit && !job.cpu_done) {
            const auto bytes = arena->record_bytes();
            job.byte_credit = true;
            stats.pending_cpu_bytes.fetch_add(bytes, std::memory_order_relaxed);
            global_pending += bytes;
        }
    }

    bool start(Handle handle, uint64_t &global_pending) {
        auto &job = jobs[handle.slot];
        if (job.active) return job.handle == handle;
        job = {};
        job.handle = handle;
        job.active = true;
        // Feature ownership is independent of any already admitted raw-KV
        // readers. Retired generations can never supply a new policy round.
        if (features) features->erase_key(handle);
        BlockView view;
        if (sample(records[handle.slot], handle.slot, view) && view.meta.handle == handle)
            job.payload = view.payload;
        stats.pending_slots.fetch_add(1, std::memory_order_relaxed);
        credit_cpu(job, global_pending);
        return true;
    }

    void cpu_complete(PendingBlock &job, uint64_t &global_pending) {
        if (job.cpu_done) return;
        job.cpu_done = true;
        if (job.byte_credit) {
            const auto bytes = arena->record_bytes();
            stats.pending_cpu_bytes.fetch_sub(bytes, std::memory_order_relaxed);
            global_pending -= bytes;
            job.byte_credit = false;
            stats.cpu_released.fetch_add(1, std::memory_order_relaxed);
        }
        records[job.handle.slot].cpu_event.store(job.handle.generation, std::memory_order_release);
        signal_completion(job.handle.slot);
    }

    bool advance_job(PendingBlock &job, uint64_t &global_pending) {
        if (!job.active) return false;
        auto &entry = records[job.handle.slot];
        const bool shutdown = closing.load(std::memory_order_acquire);
        if (job.slot_done) {
            if (shutdown || entry.slot_ack.load(std::memory_order_acquire) >= job.handle.generation) {
                job = {};
                return true;
            }
            return false;
        }
        if (!job.payload.generation) {
            BlockView view;
            if (sample(entry, job.handle.slot, view) && view.meta.handle == job.handle)
                job.payload = view.payload;
            credit_cpu(job, global_pending);
        }
        bool progress = false;
        if (job.payload.generation && !job.cpu_done) {
            const auto result = arena->retire(job.payload);
            if (result == Payload::RetireResult::Retry) stats.retries.fetch_add(1, std::memory_order_relaxed);
            if (arena->reclaimed(job.payload)) {
                cpu_complete(job, global_pending);
                progress = true;
            }
        }
        const auto result = repository->try_reclaim(job.handle, [&](const Repo::Record &) {
            // With zero metadata references, all original I/O publishers have
            // finished. Discover a late backing publication before deciding
            // this generation never owned CPU storage.
            if (!job.payload.generation) {
                BlockView view;
                if (!sample(entry, job.handle.slot, view)) return false;
                if (view.meta.handle == job.handle) job.payload = view.payload;
                credit_cpu(job, global_pending);
            }
            if (job.payload.generation && !arena->reclaimed(job.payload)) {
                arena->retire(job.payload);
                return false;
            }
            cpu_complete(job, global_pending);
            return shutdown || entry.cpu_ack.load(std::memory_order_acquire) >= job.handle.generation;
        });
        if (result == Repo::ReclaimResult::Reclaimed) {
            job.slot_done = true;
            stats.pending_slots.fetch_sub(1, std::memory_order_relaxed);
            stats.slots_released.fetch_add(1, std::memory_order_relaxed);
            entry.slot_event.store(job.handle.generation, std::memory_order_release);
            signal_completion(job.handle.slot);
            return true;
        }
        return progress;
    }

    bool release(uint64_t &global_pending) {
        bool progress = false;
        const auto count = std::min<size_t>(config.scan_blocks, jobs.size());
        for (size_t i = 0; i < count; ++i) {
            const auto slot = static_cast<uint32_t>(release_cursor);
            release_cursor = (release_cursor + 1) % jobs.size();
            auto &job = jobs[slot];
            if (shadow[slot] && !repository->try_pin({slot, shadow[slot]})) {
                shadow[slot] = 0;
                --shadow_slots;
                shadow_bytes -= arena->record_bytes();
                global_pending -= arena->record_bytes();
            }
            if (!job.active) {
                const auto retired = repository->retired_at(slot);
                if (retired.generation) {
                    const auto finalized = repository->retire_permanently(retired);
                    if (finalized == Repo::RetireResult::Committed || finalized == Repo::RetireResult::AlreadyRetired)
                        start(retired, global_pending);
                }
            }
            progress |= advance_job(job, global_pending);
        }
        return progress;
    }

    HistoricalPressure pressure(uint64_t global_pending) const {
        const auto capacity = uint64_t(arena->capacity()) * arena->record_bytes();
        const auto used = arena->stats().used_bytes;
        HistoricalPressure result;
        result.ready_slots = repository->ready_slots_for_reclaimer();
        result.low_slots = config.ready_low;
        result.high_slots = config.ready_high;
        result.pending_slots = static_cast<uint32_t>(stats.pending_slots.load(std::memory_order_relaxed)) + shadow_slots;
        result.free_cpu_bytes = capacity > used ? capacity - used : 0;
        result.low_cpu_bytes = config.cpu_free_low;
        result.high_cpu_bytes = config.cpu_free_high;
        result.pending_cpu_bytes = global_pending;
        result.requested_ready_slots = requested_ready.load(std::memory_order_acquire);
        result.scope_cpu_bytes = (stats.published.load(std::memory_order_relaxed) -
                                  stats.cpu_released.load(std::memory_order_relaxed)) * arena->record_bytes();
        result.scope_cpu_limit = requested_cpu_limit.load(std::memory_order_acquire);
        result.scope_pending_cpu_bytes = stats.pending_cpu_bytes.load(std::memory_order_relaxed) + shadow_bytes;
        return result;
    }

    HistoricalCommitCode commit_one(const HistoricalDeleteIntent &intent, uint64_t &global_pending) {
        using Code = HistoricalCommitCode;
        if (intent.incarnation != incarnation || intent.handle.slot >= jobs.size() || !intent.handle.generation)
            return Code::Stale;
        auto &job = jobs[intent.handle.slot];
        if (job.active && job.handle != intent.handle)
            return Code::Retry; // Preserve an old unacknowledged completion.
        const bool redundancy = intent.reason == HistoricalDeleteReason::Redundancy;
        if (redundancy && (!features || !intent.witness.generation || intent.witness.slot >= jobs.size() ||
                           intent.witness.slot == intent.handle.slot)) return Code::Ineligible;
        auto first = intent.handle.slot, second = intent.handle.slot;
        if (redundancy) {
            first = std::min(first, intent.witness.slot);
            second = std::max(second, intent.witness.slot);
        }
        struct PolicyUnlock {
            PublishedBlock *first = nullptr, *second = nullptr;
            ~PolicyUnlock() {
                if (second) second->policy_gate.clear(std::memory_order_release);
                if (first) first->policy_gate.clear(std::memory_order_release);
            }
        } gates;
        if (records[first].policy_gate.test_and_set(std::memory_order_acquire)) return Code::Retry;
        gates.first = &records[first];
        if (second != first) {
            if (records[second].policy_gate.test_and_set(std::memory_order_acquire)) return Code::Retry;
            gates.second = &records[second];
        }
        BlockView current;
        if (!sample(records[intent.handle.slot], intent.handle.slot, current)) return Code::Retry;
        auto live = repository->try_pin(intent.handle);
        if (!live) return repository->retired_at(intent.handle.slot) == intent.handle ? Code::AlreadyRetired : Code::Stale;
        current.meta.live = true;
        if (current.meta.handle != intent.handle) return Code::Stale;
        // This is only the proposed version for the policy's carry check.
        // The nonblocking source guard below must verify the actual version
        // before retirement. It distinguishes contention from missing keys.
        current.meta.feature_version = intent.feature_version;
        if (!policy->permits(current.meta)) return Code::Ineligible;
        Repo::Lease witness;
        if (redundancy) {
            BlockView other;
            if (!sample(records[intent.witness.slot], intent.witness.slot, other)) return Code::Retry;
            witness = repository->try_pin(intent.witness);
            if (!witness || other.meta.handle != intent.witness || !other.meta.sealed ||
                    shadow[intent.witness.slot] == intent.witness.generation) return Code::Ineligible;
        }
        // No scoring or feature copying occurs under these guards. The
        // source guard freezes question/feature identity until retirement.
        HistoricalFeatureSource::CommitGuard feature_guard;
        if (intent.reason != HistoricalDeleteReason::Lru) {
            if (!features) return Code::Ineligible;
            feature_guard = features->try_commit_guard(intent);
            if (!feature_guard.acquired) return Code::Retry;
            if (!feature_guard.valid) return Code::Ineligible;
        }
        if (config.mode == HistoricalGcMode::Shadow) {
            if (shadow[intent.handle.slot] != intent.handle.generation) {
                shadow[intent.handle.slot] = intent.handle.generation;
                ++shadow_slots;
                shadow_bytes += arena->record_bytes();
                global_pending += arena->record_bytes();
                stats.shadow_decisions.fetch_add(1, std::memory_order_relaxed);
            }
            return Code::Shadow;
        }
        switch (repository->retire_permanently(intent.handle)) {
            case Repo::RetireResult::Committed: return Code::Committed;
            case Repo::RetireResult::AlreadyRetired: return Code::AlreadyRetired;
            case Repo::RetireResult::Retry: return Code::Retry;
            default: return Code::Stale;
        }
    }

    bool commit_batch(uint64_t &global_pending) {
        using Code = HistoricalCommitCode;
        const auto &values = batch.values();
        while (batch_cursor < values.size()) {
            const auto &intent = values[batch_cursor];
            const auto code = commit_one(intent, global_pending);
            if (code == Code::Retry) {
                stats.retries.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            // Every consumed intent is reported exactly once, outside all
            // gates. This updates witness carry before the next batch item.
            policy->on_result(intent, code);
            if (code == Code::Committed || code == Code::AlreadyRetired) {
                start(intent.handle, global_pending);
                if (code == Code::Committed) {
                    stats.retired.fetch_add(1, std::memory_order_relaxed);
                    if (intent.reason == HistoricalDeleteReason::Redundancy)
                        stats.redundancy_retired.fetch_add(1, std::memory_order_relaxed);
                    if (intent.reason == HistoricalDeleteReason::LossyPressure)
                        stats.lossy_retired.fetch_add(1, std::memory_order_relaxed);
                }
            } else if (code == Code::Stale) stats.stale.fetch_add(1, std::memory_order_relaxed);
            else if (code == Code::Ineligible) stats.rejected_ineligible.fetch_add(1, std::memory_order_relaxed);
            ++batch_cursor;
        }
        batch.clear_after_ack();
        batch_cursor = 0;
        if (policy_round_done) round_active = false;
        return true;
    }

    void cancel_batch() noexcept {
        for (; batch_cursor < batch.values().size(); ++batch_cursor) {
            policy->on_result(batch.values()[batch_cursor], HistoricalCommitCode::Cancelled);
            stats.cancelled_decisions.fetch_add(1, std::memory_order_relaxed);
        }
        batch.clear_after_ack();
        batch_cursor = 0;
    }

    bool tick(uint64_t &global_pending) {
        stats.ticks.fetch_add(1, std::memory_order_relaxed);
        bool progress = release(global_pending);
        if (closing.load(std::memory_order_acquire)) {
            cancel_batch();
            const auto count = std::min<size_t>(config.scan_blocks, jobs.size());
            for (size_t i = 0; i < count; ++i) {
                const auto slot = static_cast<uint32_t>(close_cursor);
                close_cursor = (close_cursor + 1) % jobs.size();
                auto lease = repository->try_pin_slot(slot);
                if (lease) {
                    const auto result = repository->retire_permanently(lease.handle());
                    if (result == Repo::RetireResult::Committed || result == Repo::RetireResult::AlreadyRetired)
                        start(lease.handle(), global_pending);
                }
            }
            if (repository->ready_slots_for_reclaimer() == repository->capacity())
                resources_closed.store(true, std::memory_order_release);
            return progress;
        }
        if (paused.load(std::memory_order_acquire)) {
            pause_ack.store(pause_request.load(std::memory_order_acquire), std::memory_order_release);
            return progress;
        }
        try {
            if (!batch.empty()) return commit_batch(global_pending) || progress;
            if (config.mode == HistoricalGcMode::Off) return progress;
            const auto current_pressure = pressure(global_pending);
            if (!round_active) {
                if (!policy->wants_snapshot(current_pressure)) return progress;
                round_active = true;
                snapshot_ready = policy_round_done = false;
                snapshot_cursor = 0;
                snapshot.blocks.clear();
                ++snapshot.round;
            }
            if (!snapshot_ready) {
                const auto end = std::min(jobs.size(), snapshot_cursor + config.scan_blocks);
                for (; snapshot_cursor < end; ++snapshot_cursor) {
                    const auto slot = static_cast<uint32_t>(snapshot_cursor);
                    auto lease = repository->try_pin_slot(slot);
                    if (!lease) continue;
                    if (shadow[slot] == lease.handle().generation) continue;
                    BlockView view;
                    if (!sample(records[slot], slot, view) || view.meta.handle != lease.handle()) continue;
                    view.meta.live = true;
                    const auto recent = recent_begin.load(std::memory_order_acquire);
                    const auto end = history_end.load(std::memory_order_acquire);
                    if (end && lease.record().token_begin < end &&
                            lease.record().token_begin + lease.record().token_count > recent)
                        view.meta.mandatory = true;
                    view.meta.cpu_bytes = view.payload.generation ? arena->record_bytes() : 0;
                    if (features) view.meta.feature_version = features->key_version(view.meta.handle);
                    if (view.meta.gpu_resident)
                        view.meta.last_use_seq = std::max(view.meta.last_use_seq, use_stamp.load(std::memory_order_acquire));
                    snapshot.blocks.push_back(view.meta);
                }
                if (snapshot_cursor < jobs.size()) return true;
                snapshot_ready = true;
            }
            const auto result = policy->step(snapshot, current_pressure,
                                             {config.scan_blocks, config.delete_blocks}, batch);
            stats.examined.fetch_add(result.examined, std::memory_order_relaxed);
            stats.decisions.fetch_add(result.emitted, std::memory_order_relaxed);
            policy_round_done = result.status == HistoricalPolicyStatus::Complete ||
                                result.status == HistoricalPolicyStatus::NoCandidate;
            if (!batch.empty()) commit_batch(global_pending);
            else if (policy_round_done) round_active = false;
            return result.examined || result.emitted || progress;
        } catch (...) {
            // A failed strategy must not stop reclamation of earlier commits.
            cancel_batch();
            paused.store(true, std::memory_order_release);
            stats.policy_failures.fetch_add(1, std::memory_order_relaxed);
            return progress;
        }
    }
};

struct HistoricalManagerState {
    HistoricalManagerState(HistoricalGcWorker::Config c, std::shared_ptr<Payload> a)
        : config(c), arena(std::move(a)), registry(Repo::Config{c.max_scopes, 1}), scopes(c.max_scopes),
          registry_released(new std::atomic<uint64_t>[c.max_scopes]), buffers(c.max_scopes) {
        for (uint32_t i = 0; i < c.max_scopes; ++i) {
            registry_released[i].store(0, std::memory_order_relaxed);
            if (c.scope_capacity) buffers[i] = std::make_shared<HistoricalScopeStorage>(c.scope_capacity, c);
        }
        metadata_allocations.store(c.scope_capacity ? c.max_scopes : 0, std::memory_order_relaxed);
    }
    const HistoricalGcWorker::Config config;
    std::shared_ptr<Payload> arena;
    Repo registry;
    std::vector<std::shared_ptr<HistoricalScopeState>> scopes;
    std::unique_ptr<std::atomic<uint64_t>[]> registry_released;
    std::vector<std::shared_ptr<HistoricalScopeStorage>> buffers;
    std::atomic<uint64_t> metadata_allocations{0}, metadata_recycles{0};
    std::atomic<bool> closing{false}, complete{false};
    std::atomic<uint64_t> registrations{0};
    std::atomic_flag advancing = ATOMIC_FLAG_INIT;
    std::atomic_flag registering = ATOMIC_FLAG_INIT;
    std::mutex sleep_mutex;
    std::condition_variable wake;
    uint64_t pending_cpu_bytes = 0; // Sole backend owner; never allocatable credit.

    bool advance() {
        if (advancing.test_and_set(std::memory_order_acquire)) return false;
        struct Unlock { std::atomic_flag &flag; ~Unlock() { flag.clear(std::memory_order_release); } } unlock{advancing};
        bool progress = arena->advance() != 0;
        for (uint32_t slot = 0; slot < config.max_scopes; ++slot) {
            auto lease = registry.try_pin_slot(slot);
            if (lease) {
                auto &scope = scopes[slot];
                if (closing.load(std::memory_order_acquire)) scope->closing.store(true, std::memory_order_release);
                progress |= scope->tick(pending_cpu_bytes);
                if (scope->resources_closed.load(std::memory_order_acquire))
                    registry.retire_permanently(lease.handle());
            } else {
                const auto handle = registry.retired_at(slot);
                if (handle.generation) {
                    const auto result = registry.try_reclaim(handle, [&](const Repo::Record &) {
                        auto &scope = scopes[slot];
                        if (!scope) return true; // A failed scope constructor cancelled its reservation.
                        if (!scope->resources_closed.load(std::memory_order_acquire) || scope.use_count() != 1)
                            return false;
                        scope.reset(); // Large scope metadata is destroyed only here.
                        if (buffers[slot]) {
                            buffers[slot]->reset();
                            metadata_recycles.fetch_add(1, std::memory_order_relaxed);
                        }
                        return true;
                    });
                    progress |= result == Repo::ReclaimResult::Reclaimed;
                    if (result == Repo::ReclaimResult::Reclaimed)
                        registry_released[slot].store(handle.generation, std::memory_order_release);
                }
            }
        }
        return progress;
    }

    void run() {
        while (true) {
            if (config.background || closing.load(std::memory_order_acquire)) {
                for (unsigned burst = 0; burst < 16; ++burst)
                    if (!advance()) break;
            }
            if (closing.load(std::memory_order_acquire) && registry.ready_slots_for_reclaimer() == config.max_scopes &&
                    registrations.load(std::memory_order_acquire) == 0) {
                complete.store(true, std::memory_order_release);
                return;
            }
            std::unique_lock<std::mutex> lock(sleep_mutex);
            wake.wait_for(lock, std::chrono::milliseconds(config.poll_ms));
        }
    }
};

struct HistoricalGcWorker::Worker {
    explicit Worker(const std::shared_ptr<HistoricalManagerState> &state)
        : thread([state] { state->run(); }) {}
    ~Worker() { if (thread.joinable()) thread.detach(); }
    std::thread thread;
};

HistoricalGcWorker::HistoricalGcWorker(Config config, std::shared_ptr<Payload> arena) {
    validate_historical_kv_policy(config.policy, config.policy_params);
    if (!arena || !config.max_scopes || !config.poll_ms || !config.scan_blocks || !config.delete_blocks ||
            ((config.ready_low || config.ready_high) && (!config.ready_low || config.ready_low >= config.ready_high)) ||
            ((config.cpu_free_low || config.cpu_free_high) &&
             (!config.cpu_free_low || config.cpu_free_low >= config.cpu_free_high ||
              config.cpu_free_high > uint64_t(arena->capacity())*arena->record_bytes())))
        throw std::invalid_argument("Invalid historical GC worker capacity or watermarks");
    shared_ = std::make_shared<HistoricalManagerState>(config, std::move(arena));
    worker_ = std::make_unique<Worker>(shared_);
}
HistoricalGcWorker::~HistoricalGcWorker() { close(); }

std::shared_ptr<HistoricalGcScope> HistoricalGcWorker::register_scope(std::shared_ptr<Repo> repository,
                                                                    std::unique_ptr<HistoricalKvPolicy> policy) {
    shared_->registrations.fetch_add(1, std::memory_order_acq_rel);
    struct RegistrationExit {
        HistoricalManagerState &state;
        ~RegistrationExit() { state.registrations.fetch_sub(1, std::memory_order_release); }
    } registration{*shared_};
    if (shared_->closing.load(std::memory_order_acquire)) throw std::runtime_error("Historical worker is closing");
    if (!repository) throw std::invalid_argument("Historical scope needs a repository");
    if (shared_->config.scope_capacity && repository->capacity() != shared_->config.scope_capacity)
        throw std::invalid_argument("Historical scope differs from preallocated repository capacity");
    if (shared_->registering.test_and_set(std::memory_order_acquire)) return {};
    struct RegistrationUnlock {
        std::atomic_flag &flag;
        ~RegistrationUnlock() { flag.clear(std::memory_order_release); }
    } unlock{shared_->registering};
    auto reservation = shared_->registry.try_reserve(0, 1);
    if (!reservation) return {};
    const auto handle = reservation.record().handle;
    auto buffers = shared_->buffers[handle.slot];
    if (!buffers) {
        buffers = std::make_shared<HistoricalScopeStorage>(repository->capacity(), shared_->config);
        shared_->metadata_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    auto state = std::make_shared<HistoricalScopeState>(std::move(repository), shared_->arena,
                                                       shared_->config, handle.generation, std::move(policy), std::move(buffers));
    auto scope = std::shared_ptr<HistoricalGcScope>(new HistoricalGcScope(shared_, state, {}));
    shared_->scopes[handle.slot] = state;
    scope->registry_handle_ = handle;
    scope->registry_lease_ = reservation.commit();
    return scope;
}
void HistoricalGcWorker::advance() { shared_->advance(); }
bool HistoricalGcWorker::close() {
    shared_->closing.store(true, std::memory_order_release);
    shared_->wake.notify_one();
    return shared_->complete.load(std::memory_order_acquire);
}
HistoricalStorageStats HistoricalGcWorker::storage_stats() const {
    return {shared_->metadata_allocations.load(std::memory_order_relaxed),
            shared_->metadata_recycles.load(std::memory_order_relaxed),
            shared_->config.scope_capacity ? shared_->config.max_scopes : 0,
            shared_->config.scope_capacity};
}

HistoricalGcScope::HistoricalGcScope(std::shared_ptr<HistoricalManagerState> owner,
                                    std::shared_ptr<HistoricalScopeState> state, Repo::Lease lease)
    : owner_(std::move(owner)), state_(std::move(state)), registry_lease_(std::move(lease)) {}
HistoricalGcScope::~HistoricalGcScope() { close(); }
HistoricalFrameHold::HistoricalFrameHold(std::shared_ptr<HistoricalScopeState> state) : state_(std::move(state)) {}
HistoricalFrameHold::~HistoricalFrameHold() {
    for (const auto &lease : leases_)
        state_->records[lease.handle().slot].policy_holds.fetch_sub(1, std::memory_order_release);
}

std::shared_ptr<HistoricalFrameHold> HistoricalGcScope::protect(const std::vector<Handle> &handles) {
    if (!state_ || handles.empty()) return {};
    auto hold = std::shared_ptr<HistoricalFrameHold>(new HistoricalFrameHold(state_));
    hold->leases_.reserve(handles.size());
    for (auto handle : handles) {
        if (handle.slot >= state_->repository->capacity()) return {};
        auto &entry = state_->records[handle.slot];
        if (entry.policy_gate.test_and_set(std::memory_order_acquire)) return {};
        auto lease = state_->repository->try_pin(handle);
        const bool admitted = bool(lease);
        if (lease) {
            entry.policy_holds.fetch_add(1, std::memory_order_release);
            hold->leases_.push_back(std::move(lease));
        }
        entry.policy_gate.clear(std::memory_order_release);
        if (!admitted) return {};
    }
    for (const auto &lease : hold->leases_)
        if (!state_->repository->try_pin(lease.handle())) return {};
    return hold;
}
bool HistoricalGcScope::publish(const Repo::Lease &lease, const Payload::Read &cpu, uint32_t flags, uint64_t use) {
    if (!state_) throw std::runtime_error("Historical scope is closed");
    return state_->publish(lease, &cpu, flags, use);
}
bool HistoricalGcScope::update(const Repo::Lease &lease, uint32_t flags, uint64_t use) {
    if (!state_) throw std::runtime_error("Historical scope is closed");
    return state_->publish(lease, nullptr, flags, use);
}
bool HistoricalGcScope::configure_features(uint32_t layers, uint32_t kv_heads, uint32_t q_heads, uint32_t dimension) {
    if (!state_ || state_->closing.load(std::memory_order_acquire) || !state_->features) return false;
    try { state_->features->configure(layers, kv_heads, q_heads, dimension); }
    catch (const std::bad_alloc &) { return false; }
    return true;
}
std::array<size_t, 4> HistoricalGcScope::feature_geometry() const {
    return state_ && state_->features ? state_->features->geometry() : std::array<size_t, 4>{};
}
bool HistoricalGcScope::publish_features(const Repo::Lease &lease, const float *data, size_t elements) {
    if (!state_ || state_->closing.load(std::memory_order_acquire) || !state_->features) return false;
    if (!state_->repository->owns(lease)) throw std::invalid_argument("Historical features have a foreign repository lease");
    const auto handle = lease.handle();
    BlockView current;
    auto live = state_->repository->try_pin(handle);
    if (!live || !sample(state_->records[handle.slot], handle.slot, current) ||
            current.meta.handle != handle || !current.meta.sealed) return false;
    // The source makes a bounded native copy; it retains no Python buffer and
    // the longer validation/copy is outside policy gates. A final fresh pin
    // detects retirement that won while that independent copy was produced.
    if (!state_->features->publish_key(handle, data, elements)) return false;
    auto final = state_->repository->try_pin(handle);
    if (!final || state_->closing.load(std::memory_order_acquire)) {
        state_->features->erase_key(handle);
        return false;
    }
    return true;
}
uint64_t HistoricalGcScope::begin_question(uint64_t expected_tokens) {
    if (!state_ || state_->closing.load(std::memory_order_acquire) || !state_->features) return 0;
    return state_->features->begin_question(expected_tokens);
}
bool HistoricalGcScope::append_question(uint64_t epoch, uint64_t offset, const float *data,
                                       size_t elements, uint32_t tokens) {
    if (!state_ || state_->closing.load(std::memory_order_acquire) || !state_->features) return false;
    const auto width = state_->features->query_elements_per_token();
    if (!width || !tokens || width > SIZE_MAX / tokens || elements != width * tokens)
        throw std::invalid_argument("Historical question buffer has invalid geometry");
    return state_->features->append_question(epoch, offset, data, tokens);
}
bool HistoricalGcScope::observe_query(uint64_t epoch, const float *data, size_t elements, uint32_t tokens) {
    if (!state_ || state_->closing.load(std::memory_order_acquire) || !state_->features) return false;
    const auto width = state_->features->query_elements_per_token();
    if (!width || !tokens || width > SIZE_MAX / tokens || elements != width * tokens)
        throw std::invalid_argument("Historical observation buffer has invalid geometry");
    return state_->features->observe(epoch, data, tokens);
}
void HistoricalGcScope::invalidate_question() {
    if (state_ && state_->features) state_->features->invalidate_question();
}
std::map<std::string, uint64_t> HistoricalGcScope::feature_stats() const {
    return state_ && state_->features ? state_->features->stats() : std::map<std::string, uint64_t>{};
}
void HistoricalGcScope::used(uint64_t sequence) {
    if (state_) state_->use_stamp.store(sequence, std::memory_order_release);
}
void HistoricalGcScope::observe(const Repo::Lease &lease, bool sealed, bool gpu, bool mandatory) {
    const uint32_t flags = (sealed ? HistoricalSealed : 0u) | (gpu ? HistoricalGpu : 0u) |
                           (mandatory ? HistoricalMandatory : 0u);
    if (!update(lease, flags)) throw std::logic_error("Concurrent engine history publishers");
}
void HistoricalGcScope::history_changed(uint64_t end, uint64_t recent) {
    if (!state_) return;
    state_->recent_begin.store(recent, std::memory_order_release);
    state_->history_end.store(end, std::memory_order_release);
}
void HistoricalGcScope::begin_step() {
    if (state_) state_->use_stamp.fetch_add(1, std::memory_order_release);
}
bool HistoricalGcScope::observes(const std::shared_ptr<Repo> &repository) const {
    return state_ && state_->repository == repository;
}
qw3::KvMemExecutionAdmission HistoricalGcScope::admit(Handle handle) {
    if (!state_) return {};
    state_->stats.admission_attempts.fetch_add(1, std::memory_order_relaxed);
    auto retry = [&]() -> qw3::KvMemExecutionAdmission {
        state_->stats.admission_retries.fetch_add(1, std::memory_order_relaxed);
        return {};
    };
    auto initial = state_->repository->try_pin(handle);
    if (!initial) return retry();
    BlockView view;
    if (!sample(state_->records[handle.slot], handle.slot, view) || view.meta.handle != handle) return retry();
    Payload::Read read;
    if (view.payload.generation) {
        read = state_->arena->acquire(view.payload);
        if (!read) return retry();
    }
    // This final fresh pin is the admission point. If retirement won the
    // race, an earlier metadata/read lease cannot grant a new window rights.
    auto execution = state_->repository->try_pin(handle);
    if (!execution) return retry();
    return {std::move(execution), read ? std::make_shared<HistoricalReadOwner>(std::move(read)) : nullptr};
}
void HistoricalGcScope::pause_policy(bool paused) {
    if (!state_) return;
    state_->pause_request.fetch_add(1, std::memory_order_acq_rel);
    state_->paused.store(paused, std::memory_order_release);
}
bool HistoricalGcScope::policy_paused() const {
    return !state_ || (state_->paused.load(std::memory_order_acquire) &&
        state_->pause_ack.load(std::memory_order_acquire) == state_->pause_request.load(std::memory_order_acquire));
}
void HistoricalGcScope::request_capacity(uint32_t ready_slots, uint64_t retained_cpu_bytes) {
    if (!state_ || ready_slots > state_->repository->capacity())
        throw std::invalid_argument("Historical capacity request exceeds a live scope");
    state_->requested_ready.store(ready_slots, std::memory_order_release);
    state_->requested_cpu_limit.store(retained_cpu_bytes, std::memory_order_release);
}
HistoricalGcStats HistoricalGcScope::stats() const { return state_ ? state_->stats.snapshot() : final_stats_; }

std::vector<HistoricalCompletion> HistoricalGcScope::completions(size_t limit) {
    if (!limit) throw std::invalid_argument("Historical completion budget must be positive");
    std::vector<HistoricalCompletion> result;
    if (!state_) return result;
    result.reserve(std::min(limit, state_->jobs.size()));
    const auto words = std::min(state_->completion_words, limit);
    size_t examined = 0;
    for (size_t i = 0; i < words && examined < limit; ++i) {
        const auto word = completion_cursor_ / 64;
        const auto offset = completion_cursor_ % 64;
        auto bits = state_->completion_bits[word].load(std::memory_order_acquire) & (~uint64_t{0} << offset);
        while (bits && examined < limit) {
            const auto bit = static_cast<unsigned>(__builtin_ctzll(bits));
            bits &= bits - 1;
            const auto slot = static_cast<uint32_t>(word * 64 + bit);
            ++examined;
            completion_cursor_ = (slot + 1 == state_->jobs.size()) ? 0 : slot + 1;
            auto &entry = state_->records[slot];
            const auto cpu = entry.cpu_event.load(std::memory_order_acquire);
            const auto full = entry.slot_event.load(std::memory_order_acquire);
            const bool cpu_pending = cpu > entry.cpu_ack.load(std::memory_order_relaxed);
            const bool slot_pending = full > entry.slot_ack.load(std::memory_order_relaxed);
            if (cpu_pending) result.push_back({{slot, cpu}, true, slot_pending && full == cpu});
            else if (slot_pending) result.push_back({{slot, full}, false, true});
            else state_->clear_completion_if_acknowledged(slot);
        }
        if (examined == limit) break;
        completion_cursor_ = (word + 1 == state_->completion_words) ? 0 : (word + 1) * 64;
    }
    return result;
}
bool HistoricalGcScope::acknowledge(const HistoricalCompletion &event) {
    if (!state_ || event.handle.slot >= state_->jobs.size() || !event.handle.generation ||
            (!event.cpu && !event.slot)) return false;
    auto &entry = state_->records[event.handle.slot];
    if ((event.cpu && entry.cpu_event.load(std::memory_order_acquire) < event.handle.generation) ||
            (event.slot && entry.slot_event.load(std::memory_order_acquire) < event.handle.generation)) return false;
    if (event.cpu && entry.cpu_ack.load(std::memory_order_relaxed) < event.handle.generation)
        entry.cpu_ack.store(event.handle.generation, std::memory_order_release);
    if (event.slot && entry.slot_ack.load(std::memory_order_relaxed) < event.handle.generation)
        entry.slot_ack.store(event.handle.generation, std::memory_order_release);
    state_->clear_completion_if_acknowledged(event.handle.slot);
    return true;
}
bool HistoricalGcScope::close() {
    if (state_) {
        state_->closing.store(true, std::memory_order_release);
        registry_lease_.reset();
        if (!state_->resources_closed.load(std::memory_order_acquire)) return false;
        final_stats_ = state_->stats.snapshot();
        state_.reset(); // The native registry owns final metadata destruction.
    }
    // Its ACK includes registry reuse. A scheduler may immediately admit a
    // replacement instead of racing the native metadata reclaimer.
    return registry_handle_.generation &&
        owner_->registry_released[registry_handle_.slot].load(std::memory_order_acquire) >= registry_handle_.generation;
}

} // namespace vllm_kvmem
