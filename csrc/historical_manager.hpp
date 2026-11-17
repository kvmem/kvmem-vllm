// One archive-wide native policy worker; no Python callback or device work.
#pragma once

#include "historical_kv_policy.hpp"
#include "historical_policy_factory.hpp"
#include "historical_payload.hpp"
#include "qw3/kvmem_store.hpp"

#include <memory>
#include <map>
#include <array>
#include <vector>

namespace vllm_kvmem {

struct HistoricalManagerState;
struct HistoricalScopeState;

enum class HistoricalGcMode { Off, Shadow, Delete };
// Who commits retirements. Local: this worker's own native thread (TP=1).
// Tensor parallelism keeps one archive per rank and every rank must retire the
// same logical blocks at the same engine step: the leader (rank 0) only
// PROPOSEs its policy's intents, followers never run the policy, and all
// ranks commit an agreed set through the external_* scope calls.
enum class HistoricalGcCommit { Local, Propose, Follow };
enum HistoricalBlockFlag : uint32_t {
    HistoricalSealed = 1, HistoricalMandatory = 2, HistoricalGpu = 4, HistoricalCurrentWindow = 8
};

struct HistoricalGcStats {
    uint64_t published = 0, ticks = 0, examined = 0, decisions = 0;
    uint64_t retired = 0, cpu_released = 0, slots_released = 0;
    uint64_t pending_slots = 0, pending_cpu_bytes = 0;
    uint64_t retries = 0, stale = 0, policy_failures = 0, shadow_decisions = 0;
    uint64_t admission_attempts = 0, admission_retries = 0;
    uint64_t redundancy_retired = 0, lossy_retired = 0;
    uint64_t lru_fallback_retired = 0, lru_representative_retired = 0, thinking_classified = 0, thinking_pure = 0;
    uint64_t rejected_ineligible = 0, cancelled_decisions = 0;
    // Shared prefixes: records whose payload is shared, and those released
    // logically (no bytes freed). Private bytes exclude marked minus released.
    uint64_t shared_marked = 0, shared_released = 0;
    // Opt-in thinking age rule: records retired, commit-time refusals, and
    // (a gauge from the last pass) old thinking left in place because the
    // current window, recent range or a hold protected it. Retried each turn.
    uint64_t thinking_age_retired = 0, thinking_age_ineligible = 0, thinking_age_protected = 0;
};

struct HistoricalStorageStats {
    uint64_t allocations = 0, recycles = 0;
    uint32_t pooled_scopes = 0, scope_capacity = 0;
};

struct HistoricalCompletion {
    qw3::KvMemBlockHandle handle;
    bool cpu = false, slot = false;
};
struct HistoricalReadOwner final : public qw3::KvMemCpuReadOwner {
    HistoricalPayloadArena::Read read;
    explicit HistoricalReadOwner(HistoricalPayloadArena::Read value) : read(std::move(value)) {}
};
class HistoricalFrameHold {
public:
    ~HistoricalFrameHold();
    HistoricalFrameHold(const HistoricalFrameHold &) = delete;
    HistoricalFrameHold &operator=(const HistoricalFrameHold &) = delete;
private:
    friend class HistoricalGcScope;
    explicit HistoricalFrameHold(std::shared_ptr<HistoricalScopeState>, bool current = false);
    std::shared_ptr<HistoricalScopeState> state_;
    bool current_ = false;
    std::vector<qw3::KvMemRepository::Lease> leases_;
};

class HistoricalGcScope : public qw3::KvMemHistoryObserver {
public:
    ~HistoricalGcScope();
    HistoricalGcScope(const HistoricalGcScope &) = delete;
    HistoricalGcScope &operator=(const HistoricalGcScope &) = delete;
    // Publish immutable backing while the original I/O operation still holds
    // its real repository and CPU leases. Completion is legal after retirement
    // or close; it never reopens this generation for selection.
    bool publish(const qw3::KvMemRepository::Lease &, const HistoricalPayloadArena::Read &,
                 uint32_t flags, uint64_t use_seq = 0);
    bool update(const qw3::KvMemRepository::Lease &, uint32_t flags, uint64_t use_seq = 0);
    // Shared prefixes (docs/prefix-sharing.md, phase 2): this live published
    // generation's payload is shared with other sessions or a prefix pool. It
    // stops counting toward the scope's private bytes, and its retirement only
    // drops this session's reference. False when the generation is not live
    // and published.
    bool mark_shared(const qw3::KvMemRepository::Lease &);
    // Sole engine publisher; one O(1) use stamp per public model iteration.
    void used(uint64_t sequence);
    void observe(const qw3::KvMemRepository::Lease &, bool sealed, bool gpu, bool mandatory,
                 bool current_window = false) override;
    void history_changed(uint64_t end, uint64_t recent_begin) override;
    void begin_step() override;
    bool observes(const std::shared_ptr<qw3::KvMemRepository> &) const override;
    qw3::KvMemExecutionAdmission admit(qw3::KvMemBlockHandle) override;
    std::shared_ptr<HistoricalFrameHold> protect(const std::vector<qw3::KvMemBlockHandle> &);
    std::shared_ptr<void> protect_selection(const std::vector<qw3::KvMemBlockHandle> &) override;
    // Same current-window holds as protect_selection, as a typed owner for
    // engine-side frames the store does not own (layerwise per-layer windows).
    std::shared_ptr<HistoricalFrameHold> protect_current(const std::vector<qw3::KvMemBlockHandle> &);
    bool configure_features(uint32_t layers, uint32_t kv_heads, uint32_t q_heads, uint32_t dimension);
    std::array<size_t, 4> feature_geometry() const;
    bool publish_features(const qw3::KvMemRepository::Lease &, const float *, size_t elements);
    // turn: 1-based ordinal of the assistant message holding the record's
    // first token (0 = unknown); used only by the opt-in thinking age rule.
    bool publish_thinking(const qw3::KvMemRepository::Lease &, uint32_t thinking_tokens, uint32_t known_tokens,
                          uint32_t turn = 0);
    // Assistant messages complete in this session's accepted history.
    void publish_turns(uint64_t completed);
    uint64_t begin_question(uint64_t expected_tokens);
    bool append_question(uint64_t epoch, uint64_t offset, const float *, size_t elements, uint32_t tokens);
    bool observe_query(uint64_t epoch, const float *, size_t elements, uint32_t tokens);
    void invalidate_question();
    std::map<std::string, uint64_t> feature_stats() const;
    void pause_policy(bool paused);
    bool policy_paused() const;
    void request_capacity(uint32_t ready_slots, uint64_t retained_cpu_bytes = UINT64_MAX);
    // Bounded read of reliable per-slot completion state. The caller explicitly
    // acknowledges only after descriptor/index cleanup succeeds; observing a
    // completion is not itself permission to reuse its metadata.
    std::vector<HistoricalCompletion> completions(size_t limit = 64);
    bool acknowledge(const HistoricalCompletion &);
    // Synchronized commits (commit mode Propose/Follow). Each call excludes the
    // native worker for its own duration only; between prepare and release the
    // prepared records keep their policy gates (no new holds), the scope does
    // not acknowledge a pause, and their retired slots start no release job.
    // Leader only: its pending policy batch, unchanged until committed. Empty
    // while paused/closing or after pressure withdrew the batch.
    std::vector<qw3::KvMemBlockHandle> external_proposals();
    bool external_proposal_ready() const;
    // Check, without retiring, whether each handle is retirable now: live,
    // sealed, not current window/recent/held/mandatory, and on the leader the
    // strategy's own intent checks. True keeps that record prepared.
    std::vector<bool> external_prepare(const std::vector<qw3::KvMemBlockHandle> &);
    // Retire exactly these prepared handles, in order (the set every rank
    // agreed on). After release, the leader reports every batch intent to its policy.
    std::vector<bool> external_commit(const std::vector<qw3::KvMemBlockHandle> &);
    // Drop prepared state; committed slots may then start their CPU release.
    void external_release();
    HistoricalGcStats stats() const;
    bool close();
private:
    friend class HistoricalGcWorker;
    std::shared_ptr<HistoricalFrameHold> protect_impl(const std::vector<qw3::KvMemBlockHandle> &, bool current);
    HistoricalGcScope(std::shared_ptr<HistoricalManagerState>,
                      std::shared_ptr<HistoricalScopeState>, qw3::KvMemRepository::Lease);
    std::shared_ptr<HistoricalManagerState> owner_;
    std::shared_ptr<HistoricalScopeState> state_;
    qw3::KvMemRepository::Lease registry_lease_;
    qw3::KvMemBlockHandle registry_handle_;
    HistoricalGcStats final_stats_;
    size_t completion_cursor_ = 0;
};

class HistoricalGcWorker {
public:
    struct Config {
        uint32_t max_scopes = 1, poll_ms = 10;
        uint32_t scope_capacity = 0; // Serving preallocates all large scope buffers at startup.
        uint32_t ready_low = 0, ready_high = 0;
        uint64_t cpu_free_low = 0, cpu_free_high = 0;
        size_t scan_blocks = 256, delete_blocks = 64;
        HistoricalGcMode mode = HistoricalGcMode::Off;
        HistoricalGcCommit commit = HistoricalGcCommit::Local;
        // Compatibility fallback; explicit policy_params take precedence.
        bool keep_mandatory = true, background = true;
        std::string policy = "lru";
        HistoricalPolicyParameters policy_params;
    };
    HistoricalGcWorker(Config, std::shared_ptr<HistoricalPayloadArena>);
    ~HistoricalGcWorker();
    HistoricalGcWorker(const HistoricalGcWorker &) = delete;
    HistoricalGcWorker &operator=(const HistoricalGcWorker &) = delete;
    // Sole registration producer. Empty means scope capacity/admission failed.
    std::shared_ptr<HistoricalGcScope> register_scope(std::shared_ptr<qw3::KvMemRepository>,
                                                   std::unique_ptr<HistoricalKvPolicy> = {});
    // Before any scope registers (tensor-parallel role); same mode is a no-op.
    void set_commit(HistoricalGcCommit);
    HistoricalGcCommit commit() const;
    // Manual test driver; production uses the single native worker thread.
    void advance();
    bool close();
    HistoricalStorageStats storage_stats() const;
private:
    struct Worker;
    std::shared_ptr<HistoricalManagerState> shared_;
    std::unique_ptr<Worker> worker_;
};

} // namespace vllm_kvmem
