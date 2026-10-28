#include "historical_manager.hpp"
#include "historical_redundancy.hpp"

#include <array>
#include <chrono>
#include <iostream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <thread>

using namespace vllm_kvmem;
using Repo = qw3::KvMemRepository;
using Payload = HistoricalPayloadArena;

static void check(bool value, const char *why) {
    if (!value) throw std::runtime_error(why);
}

template <typename Predicate>
static void eventually(Predicate predicate, const char *why) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        check(std::chrono::steady_clock::now() < deadline, why);
        std::this_thread::yield();
    }
}

struct Fixture {
    std::shared_ptr<Payload> arena;
    std::shared_ptr<Repo> repo;
    std::unique_ptr<HistoricalGcWorker> worker;
    std::shared_ptr<HistoricalGcScope> scope;
    std::vector<Repo::Lease> gpu;
    std::vector<qw3::KvMemBlockHandle> handles;
    uint64_t payload_epoch = 0;

    Fixture(HistoricalGcMode mode = HistoricalGcMode::Delete, bool background = false,
            std::unique_ptr<HistoricalKvPolicy> policy = {}, uint32_t max_scopes = 1, bool preallocate = false,
            HistoricalPolicyParameters params = {}, std::string policy_name = "lru") {
        arena = std::make_shared<Payload>(Payload::Config{16, 16, 1, 16, 16, background});
        repo = std::make_shared<Repo>(Repo::Config{8, 4});
        HistoricalGcWorker::Config config;
        config.max_scopes = max_scopes;
        config.scope_capacity = preallocate ? 8 : 0;
        config.poll_ms = 1;
        config.ready_low = 2;
        config.ready_high = 4;
        config.scan_blocks = 8;
        config.delete_blocks = 2;
        config.mode = mode;
        config.policy = std::move(policy_name);
        config.policy_params = std::move(params);
        config.background = background;
        worker = std::make_unique<HistoricalGcWorker>(config, arena);
        scope = worker->register_scope(repo, std::move(policy));
        check(bool(scope), "scope registration failed");
    }
    void append(uint32_t flags = HistoricalSealed | HistoricalGpu, uint64_t use = 0) {
        const auto index = handles.size();
        auto lease = repo->try_reserve(index * 4, 4).commit();
        const Payload::Handle backing{static_cast<uint32_t>(index), payload_epoch + index + 1};
        std::array<uint8_t, 16> bytes{};
        bytes.fill(uint8_t(index));
        arena->publish(backing, bytes.data(), bytes.size());
        auto cpu = arena->acquire(backing);
        check(scope->publish(lease, cpu, flags, use), "backing publication failed");
        handles.push_back(lease.handle());
        gpu.push_back(std::move(lease));
    }
    void step(unsigned times = 1, bool ack = true) {
        for (unsigned i = 0; i < times; ++i) {
            worker->advance();
            arena->advance();
            if (ack) for (const auto &event : scope->completions(8)) scope->acknowledge(event);
        }
    }
    ~Fixture() {
        gpu.clear();
        if (scope) eventually([&] { worker->advance(); return scope->close(); }, "scope cleanup stalled");
        scope.reset();
        if (worker) eventually([&] { return worker->close(); }, "worker cleanup stalled");
        worker.reset();
        eventually([&] { return arena->close(); }, "arena cleanup stalled");
    }
};

static void lru_retirement_and_cpu_reclamation_do_not_wait_for_gpu() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    fixture.step(12);
    const auto stats = fixture.scope->stats();
    check(stats.retired == 4 && stats.cpu_released == 4, "LRU did not retire/release exactly the shortage");
    check(stats.pending_slots == 4 && stats.slots_released == 0, "active GPU metadata was recycled");
    check(fixture.arena->stats().used_bytes == 64, "CPU bytes remained tied to active GPU leases");
    for (unsigned i = 0; i < 4; ++i) {
        check(!fixture.repo->try_pin(fixture.handles[i]), "LRU order was not oldest first");
        check(fixture.gpu[i].record().token_begin == i * 4, "active GPU operation lost its identity");
    }
    for (unsigned i = 4; i < 8; ++i) check(bool(fixture.repo->try_pin(fixture.handles[i])), "LRU deleted beyond target");
    fixture.gpu.clear();
    fixture.step(12);
    check(fixture.repo->ready_slots() == 4 && fixture.scope->stats().pending_slots == 0,
          "last GPU completion did not return metadata slots");
}

static void completions_are_reliable_until_acknowledged_and_across_reuse() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    fixture.step(12, false);
    fixture.gpu.clear();
    fixture.step(4, false);
    check(fixture.repo->ready_slots() == 0, "repository reused a slot before CPU cleanup acknowledgement");
    auto cpu = fixture.scope->completions(8);
    check(cpu.size() == 4, "unconsumed CPU completion was lost");
    for (const auto &event : cpu) check(event.cpu && !event.slot, "unexpected completion phase");
    fixture.step(2, false);
    check(fixture.repo->ready_slots() == 0, "merely observing completion acknowledged cleanup");
    for (const auto &event : cpu) check(fixture.scope->acknowledge(event), "CPU completion acknowledgement failed");
    fixture.step(2, false);
    check(fixture.repo->ready_slots() == 4, "acknowledged CPU cleanup did not permit repository reuse");
    auto replacement = fixture.repo->try_reserve(32, 4).commit();
    const auto new_handle = replacement.handle();
    const auto old = fixture.handles[new_handle.slot];
    std::array<uint8_t, 16> bytes{};
    fixture.arena->publish({8, 9}, bytes.data(), bytes.size());
    auto payload = fixture.arena->acquire({8, 9});
    check(fixture.scope->publish(replacement, payload, HistoricalSealed | HistoricalGpu),
          "new generation could not publish before old slot acknowledgement");
    payload.reset();
    auto slots = fixture.scope->completions(8);
    bool found = false;
    for (const auto &event : slots) {
        if (event.handle == old && event.slot) found = true;
        fixture.scope->acknowledge(event);
    }
    check(found, "new generation overwrote an unconsumed old slot completion");
    check(bool(fixture.repo->try_pin(new_handle)), "old completion deleted replacement history");
    replacement.reset();
}

static void off_and_shadow_do_not_change_real_history() {
    for (const auto mode : {HistoricalGcMode::Off, HistoricalGcMode::Shadow}) {
        Fixture fixture(mode);
        for (unsigned i = 0; i < 8; ++i) fixture.append();
        fixture.step(32);
        for (const auto handle : fixture.handles) check(bool(fixture.repo->try_pin(handle)), "off/shadow hid real KV");
        check(fixture.arena->stats().used_bytes == 128, "off/shadow freed real CPU bytes");
        const auto stats = fixture.scope->stats();
        check(stats.retired == 0 && stats.cpu_released == 0, "off/shadow reported real retirement");
        check(mode == HistoricalGcMode::Off ? stats.decisions == 0 : stats.shadow_decisions == 4,
              "off/shadow policy accounting was incorrect");
    }
}

class DeletePublished final : public HistoricalKvPolicy {
public:
    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot, const HistoricalPressure &,
                                HistoricalWorkBudget, HistoricalDecisionBatch &out) override {
        if (snapshot.blocks.empty()) return {HistoricalPolicyStatus::NoCandidate, 0, 0};
        out.push({snapshot.blocks.front().handle, snapshot.incarnation, snapshot.round});
        return {HistoricalPolicyStatus::Complete, 1, 1};
    }
};

static void proactive_policy_can_delete_unsealed_active_blocks_without_pressure() {
    Fixture fixture(HistoricalGcMode::Delete, false, std::make_unique<DeletePublished>());
    fixture.append(HistoricalGpu | HistoricalMandatory);
    fixture.step(8);
    check(fixture.scope->stats().retired == 1 && fixture.scope->stats().cpu_released == 1,
          "manager vetoed a custom strategy by watermarks, active, sealed or mandatory flags");
    check(fixture.gpu[0].record().token_begin == 0, "proactive deletion interrupted old operation");
}

class ThrowAfterOne final : public HistoricalKvPolicy {
    bool emitted_ = false;
public:
    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot, const HistoricalPressure &,
                                HistoricalWorkBudget, HistoricalDecisionBatch &out) override {
        if (emitted_) throw std::runtime_error("injected policy failure");
        emitted_ = true;
        out.push({snapshot.blocks.front().handle, snapshot.incarnation, snapshot.round});
        return {HistoricalPolicyStatus::Complete, 1, 1};
    }
};

static void policy_failure_and_pause_do_not_stop_committed_releases() {
    Fixture fixture(HistoricalGcMode::Delete, false, std::make_unique<ThrowAfterOne>());
    fixture.append();
    fixture.step(8);
    check(fixture.scope->stats().policy_failures == 1, "policy exception was not surfaced");
    check(fixture.scope->stats().cpu_released == 1, "policy failure prevented CPU reclamation");
    fixture.scope->pause_policy(true);
    fixture.gpu.clear();
    fixture.step(8);
    check(fixture.scope->stats().slots_released == 1, "paused strategy prevented old metadata release");
}

static void original_io_can_publish_after_retirement_without_resurrection() {
    Fixture fixture(HistoricalGcMode::Off);
    auto io = fixture.repo->try_reserve(0, 4).commit();
    const auto handle = io.handle();
    check(fixture.repo->retire_permanently(handle) == Repo::RetireResult::Committed, "retirement failed");
    fixture.step(2);
    std::array<uint8_t, 16> bytes{};
    fixture.arena->publish({0, 1}, bytes.data(), bytes.size());
    auto cpu = fixture.arena->acquire({0, 1});
    check(fixture.scope->publish(io, cpu, HistoricalSealed), "late I/O could not register its real ownership");
    cpu.reset();
    fixture.step(8);
    check(!fixture.repo->try_pin(handle) && fixture.arena->stats().used_bytes == 0,
          "late publication reopened retired history or leaked CPU ownership");
    check(fixture.repo->ready_slots() == 7, "in-flight original I/O was freed early");
    io.reset();
    fixture.step(8);
    check(fixture.repo->ready_slots() == 8, "late I/O completion did not release its metadata");
}

static void foreign_owners_and_overwrites_are_rejected_and_scope_slots_recycle() {
    Fixture fixture(HistoricalGcMode::Off);
    fixture.append();
    auto other = std::make_shared<Repo>(Repo::Config{8, 4});
    auto foreign = other->try_reserve(0, 4).commit();
    auto cpu = fixture.arena->acquire({0, 1});
    bool rejected = false;
    try { fixture.scope->publish(foreign, cpu, HistoricalSealed); }
    catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "foreign repository with identical handle was accepted");
    check(fixture.scope->publish(fixture.gpu[0], cpu, HistoricalSealed), "idempotent backing registration failed");
    check(fixture.scope->stats().published == 1, "repeated publication double-counted backing");
    cpu.reset();
    fixture.gpu.clear();
    eventually([&] { fixture.worker->advance(); return fixture.scope->close(); }, "scope close stalled");
    fixture.scope.reset();
    fixture.worker->advance();
    fixture.scope = fixture.worker->register_scope(fixture.repo);
    check(bool(fixture.scope), "fixed scope registry did not recycle a completed namespace");
}

static void background_manager_progresses_without_foreground_policy_calls() {
    Fixture fixture(HistoricalGcMode::Delete, true);
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    eventually([&] { return fixture.scope->stats().cpu_released == 4; }, "native background manager made no CPU progress");
    check(fixture.repo->ready_slots() == 0, "background manager released active GPU metadata");
    fixture.gpu.clear();
    eventually([&] {
        for (const auto &event : fixture.scope->completions(8)) fixture.scope->acknowledge(event);
        return fixture.scope->stats().slots_released == 4;
    },
               "native background manager made no metadata progress");
}

static void store_can_finish_and_stage_out_a_retired_gpu_window() {
    Fixture fixture;
    qw3::KvMemStoreConfig config;
    config.block_tokens = 4;
    config.select_budget = 16;
    config.gen_budget = 16;
    config.sink_blocks = 1;
    config.recent_blocks = 1;
    config.optimize_stage_in = true;
    auto store = std::make_unique<qw3::KvMemStore>(config, fixture.repo);
    store->attach_history_observer(fixture.scope);
    auto append = store->prepare_repository_append(32, 0);
    check(bool(append), "store append reservation failed");
    append->commit(32);
    append.reset();
    for (unsigned i = 0; i < 8; ++i) {
        auto io = fixture.repo->try_pin_slot(i);
        std::array<uint8_t, 16> bytes{};
        fixture.arena->publish({i, i + 1}, bytes.data(), bytes.size());
        auto cpu = fixture.arena->acquire({i, i + 1});
        check(fixture.scope->publish(io, cpu, 0), "I/O backing publication failed");
        fixture.handles.push_back(io.handle());
    }
    store->gc_begin_step();
    fixture.step(16);
    check(fixture.scope->stats().cpu_released == 4 && fixture.scope->stats().slots_released == 0,
          "CPU reclamation interfered with Store GPU execution rights");
    check(bool(fixture.repo->try_pin(fixture.handles[0])) && bool(fixture.repo->try_pin(fixture.handles[7])),
          "LRU ignored Store sink/recent policy metadata");
    bool rejected = false;
    try { store->set_repository_selection({fixture.handles[1]}); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected, "old GPU ownership granted a retired block new selection rights");
    auto plan = store->set_repository_selection({fixture.handles[0], fixture.handles[5],
                                                fixture.handles[6], fixture.handles[7]});
    check(plan.stage_out.size() == 4 && plan.total_window_tokens == 16,
          "normal refresh failed to include retired old GPU pages in stage-out");
    for (auto slot : plan.stage_out)
        store->set_block_tier(slot, qw3::KvTier::SSD, -1, -1, fixture.handles[slot].generation);
    fixture.step(4);
    check(fixture.scope->stats().slots_released == 0, "previous-window fence was bypassed");
    plan.repository_previous_window.reset();
    fixture.step(8);
    check(fixture.scope->stats().slots_released == 4, "ordinary refresh did not release retired GPU generations");
    store.reset();
}

static void engine_metadata_does_not_contend_with_io_publication() {
    Fixture fixture(HistoricalGcMode::Off);
    fixture.append();
    auto cpu = fixture.arena->acquire({0, 1});
    std::thread io([&] {
        for (unsigned i = 0; i < 10000; ++i)
            check(fixture.scope->publish(fixture.gpu[0], cpu, HistoricalSealed), "sole I/O publisher contended");
    });
    for (unsigned i = 0; i < 10000; ++i) {
        fixture.scope->used(i);
        check(fixture.scope->update(fixture.gpu[0], HistoricalSealed | HistoricalGpu),
              "engine metadata acquired the I/O publication guard");
    }
    io.join();
    cpu.reset();
}

static void execution_admission_races_retirement_and_generation_reuse() {
    Fixture fixture(HistoricalGcMode::Off);
    for (uint64_t generation = 1; generation <= 500; ++generation) {
        auto gpu = fixture.repo->try_reserve(generation * 4, 4).commit();
        const auto handle = gpu.handle();
        std::array<uint8_t, 16> bytes{};
        bytes.fill(uint8_t(generation));
        fixture.arena->publish({0, generation}, bytes.data(), bytes.size());
        auto cpu = fixture.arena->acquire({0, generation});
        check(fixture.scope->publish(gpu, cpu, HistoricalSealed | HistoricalGpu), "race publication failed");
        cpu.reset();
        std::thread deletion([&] {
            while (fixture.repo->retire_permanently(handle) == Repo::RetireResult::Retry) std::this_thread::yield();
        });
        auto admission = fixture.scope->admit(handle);
        deletion.join();
        fixture.step(4);
        if (admission.lease) {
            auto source = std::dynamic_pointer_cast<HistoricalReadOwner>(admission.cpu);
            check(bool(source) && source->read.data()[0] == uint8_t(generation), "admission lost its actual CPU owner");
            check(fixture.arena->stats().used_bytes == 16, "admitted operation's CPU bytes were freed");
        }
        check(!fixture.scope->admit(handle).lease, "retired generation obtained a new execution admission");
        admission = {};
        gpu.reset();
        fixture.step(8);
        check(fixture.repo->ready_slots() == 8 && fixture.arena->stats().used_bytes == 0,
              "admission race leaked CPU or repository capacity");
    }
}

static void policy_frame_holds_revalidate_old_decisions_and_release_without_cpu_pins() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    fixture.step(); // The immutable heap already contains all eight candidates.
    auto frame = fixture.scope->protect({fixture.handles[2], fixture.handles[3]});
    check(bool(frame), "checkpoint policy admission failed");
    fixture.step(24);
    check(fixture.scope->stats().retired == 4, "revalidation did not find replacement LRU candidates");
    for (auto i : {2, 3, 6, 7}) check(bool(fixture.repo->try_pin(fixture.handles[i])), "LRU deleted a protected frame");
    for (auto i : {0, 1, 4, 5}) check(!fixture.repo->try_pin(fixture.handles[i]), "LRU order ignored delayed eligibility");
    check(fixture.arena->stats().readers == 0, "policy retention acquired CPU read rights");
    frame.reset();
}

static void policy_frame_hold_is_optional_for_the_retirement_mechanism() {
    Fixture fixture(HistoricalGcMode::Delete, false, std::make_unique<DeletePublished>());
    fixture.append();
    auto frame = fixture.scope->protect({fixture.handles[0]});
    check(bool(frame), "custom-policy frame admission failed");
    fixture.step(8);
    check(fixture.scope->stats().cpu_released == 1, "mechanism vetoed a custom policy by checkpoint ownership");
    check(!fixture.scope->protect({fixture.handles[0]}), "retired checkpoint obtained new protection rights");
    frame.reset();
}

static void failed_frame_admission_rolls_back_earlier_policy_holds() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    auto invalid = fixture.handles[1];
    ++invalid.generation;
    check(!fixture.scope->protect({fixture.handles[0], invalid}), "stale frame was admitted");
    fixture.step(16);
    check(!fixture.repo->try_pin(fixture.handles[0]), "partial failed admission leaked a policy hold");
}

static void requested_capacity_can_exceed_static_lru_watermarks() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    fixture.scope->request_capacity(6);
    fixture.step(24);
    check(fixture.scope->stats().retired == 6 && fixture.scope->stats().cpu_released == 6,
          "waiting turn could not request a reserve larger than the static high watermark");
    check(fixture.repo->ready_slots() == 0, "capacity demand bypassed the old GPU fence");
    fixture.gpu.clear();
    fixture.step(8);
    check(fixture.repo->ready_slots() == 6, "fenced capacity demand did not become ready supply");
    fixture.scope->request_capacity(0);
    fixture.step(8);
    check(fixture.scope->stats().retired == 6, "cleared admission demand continued deleting history");
}

static void scope_cpu_target_is_independent_of_global_free_bytes() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    auto frame = fixture.scope->protect({fixture.handles[0]});
    fixture.scope->request_capacity(0, 32);
    fixture.step(24);
    check(fixture.arena->stats().used_bytes == 32 && fixture.scope->stats().cpu_released == 6,
          "per-scope CPU demand did not stop at its requested retained size");
    check(bool(fixture.repo->try_pin(fixture.handles[0])), "CPU reservation deleted an LRU mandatory checkpoint");
    fixture.scope->request_capacity(0);
    fixture.step(8);
    check(fixture.scope->stats().retired == 6, "reset CPU demand was not cleared");
    frame.reset();
}

static void serving_metadata_is_preallocated_and_recycled_before_scope_ack(bool keep_mandatory = true) {
    Fixture fixture(HistoricalGcMode::Delete, false, {}, 1, true, {{"keep_mandatory", keep_mandatory}});
    check(fixture.worker->storage_stats().allocations == 1, "serving metadata was not allocated at startup");
    for (unsigned round = 0; round < 10; ++round) {
        fixture.payload_epoch = uint64_t(round)*8;
        for (unsigned i = 0; i < 8; ++i)
            fixture.append(HistoricalSealed | HistoricalGpu | (keep_mandatory ? 0 : HistoricalMandatory));
        fixture.step(16);
        check(fixture.scope->stats().retired == 4, "reused LRU policy inherited its previous incarnation");
        fixture.gpu.clear();
        eventually([&] { fixture.worker->advance(); return fixture.scope->close(); }, "pooled metadata close did not finish");
        const auto storage = fixture.worker->storage_stats();
        check(storage.allocations == 1 && storage.recycles == round+1,
              "scope registration allocated large metadata or its ACK preceded recycling");
        fixture.scope.reset();
        fixture.handles.clear();
        fixture.repo = std::make_shared<Repo>(Repo::Config{8, 4});
        fixture.scope = fixture.worker->register_scope(fixture.repo);
        check(bool(fixture.scope), "acknowledged pooled scope was not immediately reusable");
    }
}

class ScriptedDecisions final : public HistoricalKvPolicy {
public:
    std::function<void(const HistoricalSnapshot &, HistoricalDecisionBatch &)> prepare;
    std::vector<HistoricalCommitCode> results;
    std::shared_ptr<const HistoricalFeatureSource> features;
    std::optional<HistoricalFeatureSource::CommitGuard> held_guard;
    bool emitted = false, fail_after_emit = false, permit = true;
    ScriptedDecisions() { results.reserve(8); }
    void bind_features(std::shared_ptr<const HistoricalFeatureSource> source) override { features = std::move(source); }
    bool wants_snapshot(const HistoricalPressure &) override { return !emitted; }
    bool permits(const HistoricalBlockMeta &meta) const override { return permit && meta.live && !meta.mandatory; }
    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot, const HistoricalPressure &,
                                HistoricalWorkBudget, HistoricalDecisionBatch &out) override {
        emitted = true;
        prepare(snapshot, out);
        if (fail_after_emit) throw std::runtime_error("injected decision failure");
        return {HistoricalPolicyStatus::Complete, snapshot.blocks.size(), out.values().size()};
    }
    void on_result(const HistoricalDeleteIntent &, HistoricalCommitCode code) noexcept override {
        results.push_back(code);
    }
};

static void terminal_feedback_is_exact_and_shadow_is_not_a_real_commit() {
    for (const auto mode : {HistoricalGcMode::Delete, HistoricalGcMode::Shadow}) {
        auto policy = std::make_unique<ScriptedDecisions>();
        auto *script = policy.get();
        script->prepare = [](const HistoricalSnapshot &s, HistoricalDecisionBatch &out) {
            out.push({s.blocks[0].handle, s.incarnation, s.round});
            out.push({s.blocks[0].handle, s.incarnation, s.round});
        };
        Fixture fixture(mode, false, std::move(policy));
        fixture.append();
        fixture.step(8);
        check(script->results.size() == 2, "terminal intent was lost or acknowledged twice");
        if (mode == HistoricalGcMode::Delete) {
            check(script->results[0] == HistoricalCommitCode::Committed &&
                  script->results[1] == HistoricalCommitCode::AlreadyRetired, "duplicate retirement feedback was incorrect");
            check(fixture.scope->stats().retired == 1, "duplicate retirement double-counted bytes");
        } else {
            check(script->results[0] == HistoricalCommitCode::Shadow && script->results[1] == HistoricalCommitCode::Shadow,
                  "shadow result manufactured a real commit");
            check(fixture.scope->stats().retired == 0 && fixture.scope->stats().shadow_decisions == 1,
                  "duplicate shadow feedback miscounted simulated retirement");
        }
    }
}

static void cancelled_stale_and_ineligible_decisions_have_feedback() {
    for (unsigned kind = 0; kind < 3; ++kind) {
        auto policy = std::make_unique<ScriptedDecisions>();
        auto *script = policy.get();
        script->fail_after_emit = kind == 0;
        script->permit = kind != 2;
        script->prepare = [kind](const HistoricalSnapshot &s, HistoricalDecisionBatch &out) {
            auto handle = s.blocks[0].handle;
            if (kind == 1) ++handle.generation;
            out.push({handle, s.incarnation, s.round});
        };
        Fixture fixture(HistoricalGcMode::Delete, false, std::move(policy));
        fixture.append();
        fixture.step(8);
        const HistoricalCommitCode expected[] = {HistoricalCommitCode::Cancelled, HistoricalCommitCode::Stale,
                                                HistoricalCommitCode::Ineligible};
        check(script->results.size() == 1 && script->results[0] == expected[kind], "rejected intent lacked terminal feedback");
        check(fixture.scope->stats().retired == 0, "rejected intent retired CPU history");
    }
}

static uint64_t prepare_feature_pair(Fixture &fixture) {
    check(fixture.scope->configure_features(1, 1, 1, 2), "feature source was not configured");
    fixture.append(); fixture.append();
    const float key[] = {1.f, 0.f};
    check(fixture.scope->publish_features(fixture.gpu[0], key, 2) &&
          fixture.scope->publish_features(fixture.gpu[1], key, 2), "sealed features did not publish");
    const auto epoch = fixture.scope->begin_question(1);
    check(fixture.scope->append_question(epoch, 0, key, 2, 1), "complete question did not publish");
    return epoch;
}

static HistoricalDeleteIntent witnessed(const HistoricalSnapshot &s, uint64_t epoch) {
    return {s.blocks[0].handle, s.incarnation, s.round, HistoricalDeleteReason::Redundancy,
            s.blocks[0].feature_version, epoch, s.blocks[1].handle, s.blocks[1].feature_version};
}

static void redundancy_commit_rechecks_question_witness_and_protection() {
    for (unsigned kind = 0; kind < 6; ++kind) {
        auto policy = std::make_unique<ScriptedDecisions>();
        auto *script = policy.get();
        Fixture fixture(HistoricalGcMode::Delete, false, std::move(policy), 1, false, {}, "redundancy");
        const auto epoch = prepare_feature_pair(fixture);
        std::shared_ptr<HistoricalFrameHold> protected_frame;
        script->prepare = [&](const HistoricalSnapshot &s, HistoricalDecisionBatch &out) {
            auto intent = witnessed(s, epoch);
            if (kind == 1) fixture.scope->invalidate_question();
            if (kind == 2) fixture.repo->retire_permanently(intent.witness);
            if (kind == 3) protected_frame = fixture.scope->protect({intent.handle});
            if (kind == 4) ++intent.witness_feature_version;
            if (kind == 5) std::const_pointer_cast<HistoricalFeatureSource>(script->features)->erase_key(intent.handle);
            out.push(intent);
        };
        fixture.step(8);
        check(script->results.size() == 1, "witnessed intent did not finish exactly once");
        check(script->results[0] == (kind ? HistoricalCommitCode::Ineligible : HistoricalCommitCode::Committed),
              "question, witness, feature version or protection changed without rejecting intent");
        check(bool(fixture.repo->try_pin(fixture.handles[0])) == (kind != 0), "invalid witness removed victim");
        if (!kind) check(fixture.scope->stats().redundancy_retired == 1, "redundancy reason was not counted");
        protected_frame.reset();
    }
}

static void contended_feature_commit_retries_without_consuming_the_intent() {
    auto policy = std::make_unique<ScriptedDecisions>();
    auto *script = policy.get();
    Fixture fixture(HistoricalGcMode::Delete, false, std::move(policy), 1, false, {}, "redundancy");
    const auto epoch = prepare_feature_pair(fixture);
    script->prepare = [&](const HistoricalSnapshot &s, HistoricalDecisionBatch &out) {
        const auto intent = witnessed(s, epoch);
        script->held_guard.emplace(script->features->try_commit_guard(intent));
        check(script->held_guard->valid, "test feature identity was invalid");
        out.push(intent);
    };
    fixture.step();
    check(script->results.empty() && fixture.scope->stats().retired == 0, "Retry consumed an uncommitted intent");
    script->held_guard.reset();
    fixture.step(8);
    check(script->results.size() == 1 && script->results[0] == HistoricalCommitCode::Committed,
          "retry did not preserve exact witnessed intent");
}

static void features_require_live_sealed_backing_and_never_resurrect_history() {
    Fixture fixture(HistoricalGcMode::Off, false, {}, 1, false, {}, "redundancy");
    check(fixture.scope->configure_features(1, 1, 1, 2), "feature configuration failed");
    fixture.append(HistoricalGpu);
    const float key[] = {1.f, 0.f};
    check(!fixture.scope->publish_features(fixture.gpu[0], key, 2), "unsealed features were published");
    check(fixture.scope->update(fixture.gpu[0], HistoricalSealed | HistoricalGpu), "engine seal failed");
    check(fixture.scope->publish_features(fixture.gpu[0], key, 2), "sealed feature publication failed");
    check(fixture.scope->publish_features(fixture.gpu[0], key, 2), "idempotent sealed feature publication failed");
    const float changed[] = {0.f, 1.f};
    check(!fixture.scope->publish_features(fixture.gpu[0], changed, 2), "immutable generation accepted changed features");
    fixture.repo->retire_permanently(fixture.handles[0]);
    check(!fixture.scope->publish_features(fixture.gpu[0], key, 2), "retired feature publication resurrected history");
    fixture.step(8);
    check(fixture.scope->feature_stats().at("keys") == 0, "retired feature cache retained dead generations");
}

static void pooled_redundancy_scope_discards_features_question_and_carry_between_owners() {
    Fixture fixture(HistoricalGcMode::Delete, false, {}, 1, true, {}, "redundancy");
    uint64_t previous_epoch = 0;
    const auto index_bytes = fixture.scope->feature_stats().at("feature_bytes");
    for (unsigned owner = 0; owner < 3; ++owner) {
        const auto fresh = fixture.scope->feature_stats();
        check(fresh.at("keys") == 0 && fresh.at("question_ready") == 0 &&
              fresh.at("feature_bytes") == index_bytes && fresh.at("query_bytes") == 0,
              "scope pool retained charged features or an old owner's complete question");
        check(fixture.scope->feature_geometry() == std::array<size_t, 4>{}, "scope reused old feature geometry");
        check(fixture.scope->configure_features(1, 1, 1, 2), "pooled source configuration failed");
        fixture.payload_epoch = uint64_t(owner) * 8;
        const float key[] = {1.f, 0.f};
        for (unsigned i = 0; i < 8; ++i) {
            fixture.append();
            check(fixture.scope->publish_features(fixture.gpu.back(), key, 2), "new owner feature publication failed");
        }
        fixture.step(16);
        check(fixture.scope->stats().retired == 0, "new owner inherited a complete old query");
        const auto epoch = fixture.scope->begin_question(1);
        check(epoch > previous_epoch, "pooled source reused an old question epoch");
        previous_epoch = epoch;
        check(fixture.scope->append_question(epoch, 0, key, 2, 1), "new owner query publication failed");
        fixture.step(64);
        check(fixture.scope->stats().redundancy_retired == 4, "new owner retained old redundancy carry or pressure");
        fixture.gpu.clear();
        eventually([&] { fixture.worker->advance(); return fixture.scope->close(); }, "redundancy pool close did not finish");
        const auto storage = fixture.worker->storage_stats();
        check(storage.allocations == 1 && storage.recycles == owner + 1, "source pool storage was not recycled before close ACK");
        fixture.scope.reset();
        fixture.handles.clear();
        fixture.repo = std::make_shared<Repo>(Repo::Config{8, 4});
        fixture.scope = fixture.worker->register_scope(fixture.repo);
        check(bool(fixture.scope), "recycled redundancy scope could not admit new owner");
    }
}

int main() {
    try {
        lru_retirement_and_cpu_reclamation_do_not_wait_for_gpu();
        completions_are_reliable_until_acknowledged_and_across_reuse();
        off_and_shadow_do_not_change_real_history();
        proactive_policy_can_delete_unsealed_active_blocks_without_pressure();
        policy_failure_and_pause_do_not_stop_committed_releases();
        original_io_can_publish_after_retirement_without_resurrection();
        foreign_owners_and_overwrites_are_rejected_and_scope_slots_recycle();
        background_manager_progresses_without_foreground_policy_calls();
        store_can_finish_and_stage_out_a_retired_gpu_window();
        engine_metadata_does_not_contend_with_io_publication();
        execution_admission_races_retirement_and_generation_reuse();
        policy_frame_holds_revalidate_old_decisions_and_release_without_cpu_pins();
        policy_frame_hold_is_optional_for_the_retirement_mechanism();
        failed_frame_admission_rolls_back_earlier_policy_holds();
        requested_capacity_can_exceed_static_lru_watermarks();
        scope_cpu_target_is_independent_of_global_free_bytes();
        serving_metadata_is_preallocated_and_recycled_before_scope_ack();
        serving_metadata_is_preallocated_and_recycled_before_scope_ack(false);
        terminal_feedback_is_exact_and_shadow_is_not_a_real_commit();
        cancelled_stale_and_ineligible_decisions_have_feedback();
        redundancy_commit_rechecks_question_witness_and_protection();
        contended_feature_commit_retries_without_consuming_the_intent();
        features_require_live_sealed_backing_and_never_resurrect_history();
        pooled_redundancy_scope_discards_features_question_and_carry_between_owners();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Historical KV native manager tests passed\n";
}
