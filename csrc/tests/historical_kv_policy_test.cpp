#include "historical_kv_policy.hpp"
#include "historical_policy_factory.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace vllm_kvmem;
using Repository = qw3::KvMemRepository;

static void check(bool value, const char *why) {
    if (!value) throw std::runtime_error(why);
}

static HistoricalBlockMeta block(uint32_t slot, uint64_t birth, uint64_t use) {
    HistoricalBlockMeta b;
    b.handle = {slot, 1};
    b.logical_id = slot + 1;
    b.birth_seq = birth;
    b.last_use_seq = use;
    b.cpu_bytes = 100;
    b.live = b.sealed = true;
    return b;
}

static void irreversible_retirement() {
    Repository repo({1, 32});
    auto gpu = repo.try_reserve(0, 32).commit();
    auto cpu = gpu.clone();
    const auto old = gpu.handle();
    check(repo.retire_permanently(old) == Repository::RetireResult::Committed,
          "nonzero references prevented permanent retirement");
    check(!repo.try_pin(old), "permanently deleted generation accepted a new reader");
    check(!repo.rescue(gpu), "old active lease resurrected permanent deletion");
    check(repo.retire_permanently(old) == Repository::RetireResult::AlreadyRetired,
          "repeated retirement was not idempotent");
    auto transfer = cpu.clone();
    check(transfer.record().token_begin == 0, "existing operation lost its payload identity");
    unsigned releases = 0;
    auto release = [&](const Repository::Record &) { ++releases; return true; };
    cpu.reset(); transfer.reset();
    check(repo.try_reclaim(old, release) == Repository::ReclaimResult::DeferredReaders && !releases,
          "metadata slot was reused during an active GPU operation");
    gpu.reset();
    check(!releases, "last foreground reference ran the release callback");
    std::thread worker([&] {
        check(repo.try_reclaim(old, release) == Repository::ReclaimResult::Reclaimed,
              "background reclaimer could not recycle deleted generation");
    });
    worker.join();
    check(releases == 1 && repo.ready_slots() == 1, "deleted slot was not returned exactly once");
    auto fresh = repo.try_reserve(32, 32).commit();
    check(repo.retire_permanently(old) == Repository::RetireResult::StaleHandle,
          "stale intent retired a reused slot");
    check(bool(repo.try_pin(fresh.handle())), "stale deletion hid fresh history");
    check(repo.retire_permanently({99, 1}) == Repository::RetireResult::StaleHandle,
          "out of range intent was accepted");
}

static void legacy_retirement_can_be_finalized() {
    Repository repo({1, 32});
    auto active = repo.try_reserve(0, 32).commit();
    check(repo.retire(active.handle()), "legacy retirement failed");
    check(repo.retire_permanently(active.handle()) == Repository::RetireResult::Committed,
          "legacy retirement could not be made irreversible");
    check(!repo.rescue(active), "legacy rescue reopened finalized history");
}

static void stale_intents_racing_with_slot_reuse_do_not_delete_new_generations() {
    constexpr uint32_t capacity = 4;
    constexpr uint64_t iterations = 2000;
    Repository repo({capacity, 32});
    std::array<std::atomic<uint64_t>, capacity> old_generations{};
    std::atomic<bool> stop{false}, corrupt{false};
    std::thread reclaimer([&] {
        while (!stop.load(std::memory_order_acquire)) {
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                const auto h = repo.retired_at(slot);
                if (h.generation) repo.try_reclaim(h, [](const auto &) { return true; });
            }
            std::this_thread::yield();
        }
    });
    std::thread stale([&] {
        while (!stop.load(std::memory_order_acquire)) {
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                const auto old = old_generations[slot].load(std::memory_order_acquire);
                if (old && repo.retire_permanently({slot, old}) == Repository::RetireResult::Committed)
                    corrupt.store(true);
            }
        }
    });
    struct Join {
        std::atomic<bool> &stop;
        std::thread &a, &b;
        ~Join() { stop.store(true, std::memory_order_release); a.join(); b.join(); }
    } join{stop, reclaimer, stale};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (uint64_t i = 0; i < iterations; ++i) {
        Repository::Reservation reservation;
        while (!(reservation = repo.try_reserve(i*32, 32))) {
            check(std::chrono::steady_clock::now() < deadline, "stale intent test ran out of ready slots");
            std::this_thread::yield();
        }
        auto active = reservation.commit();
        Repository::RetireResult result;
        do { result = repo.retire_permanently(active.handle()); }
        while (result == Repository::RetireResult::Retry && std::chrono::steady_clock::now() < deadline);
        check(result == Repository::RetireResult::Committed, "fresh generation could not be retired");
        old_generations[active.handle().slot].store(active.handle().generation, std::memory_order_release);
    }
    while (repo.stats().reclaimed != iterations) {
        check(std::chrono::steady_clock::now() < deadline, "stale intent test did not drain");
        std::this_thread::yield();
    }
    check(!corrupt, "stale intent crossed a generation boundary during reuse");
}

static void policy_only_emits_intents() {
    HistoricalSnapshot s{7, 1, {block(0, 1, 4), block(1, 2, 2), block(2, 3, 3)}};
    s.blocks[1].gpu_resident = true;
    s.blocks[1].references = 8;
    HistoricalLruPolicy policy(8);
    HistoricalDecisionBatch out(2);
    HistoricalPressure pressure{0, 1, 2, 0};
    check(policy.step(s, pressure, {1, 1}, out).status == HistoricalPolicyStatus::Yielded,
          "policy failed to yield while scanning");
    policy.step(s, pressure, {1, 1}, out);
    check(policy.step(s, pressure, {1, 1}, out).emitted == 1, "policy lost oldest candidate");
    check(out.values()[0].handle.slot == 1, "GPU residency or references changed LRU order");
    check(s.blocks[1].live, "policy modified repository visibility");
    check(policy.step(s, pressure, {8, 8}, out).status == HistoricalPolicyStatus::Backpressure,
          "unacknowledged deletion batch was overwritten");
    out.clear_after_ack();
    pressure.pending_slots = 1;
    check(policy.step(s, pressure, {8, 8}, out).emitted == 1 &&
          out.values()[0].handle.slot == 2, "next batch did not retain candidate order");
    out.clear_after_ack();
    pressure.pending_slots = 2;
    check(policy.step(s, pressure, {8, 8}, out).status == HistoricalPolicyStatus::Complete,
          "pending slot credit did not stop excessive deletion");
}

static void byte_pressure_is_independent_of_slots() {
    HistoricalSnapshot s{7, 1, {block(0, 1, 1), block(1, 2, 2), block(2, 3, 3)}};
    HistoricalPressure pressure;
    pressure.ready_slots = 8;
    pressure.low_slots = 1;
    pressure.high_slots = 2;
    pressure.low_cpu_bytes = 100;
    pressure.high_cpu_bytes = 250;
    HistoricalLruPolicy policy(8);
    HistoricalDecisionBatch out(8);
    check(policy.step(s, pressure, {8, 8}, out).emitted == 3,
          "free metadata slots hid CPU byte pressure");
    out.clear_after_ack();
    ++s.round;
    pressure.pending_cpu_bytes = 250;
    check(policy.step(s, pressure, {8, 8}, out).emitted == 0,
          "pending CPU bytes were not deducted");
}

// Shared prefixes: shared records free no bytes, so a scope byte budget is
// met with private records only; a slot deficit may still take shared ones.
static void shared_records_count_only_against_slots() {
    auto shared = [](uint32_t slot, uint64_t age) {
        auto b = block(slot, age, age);
        b.shared = true;
        b.cpu_bytes = 0;
        return b;
    };
    HistoricalSnapshot s{7, 1, {shared(0, 1), shared(1, 2), block(2, 3, 3), block(3, 4, 4)}};
    HistoricalPressure pressure;
    pressure.ready_slots = 8;
    pressure.scope_cpu_bytes = 400;       // two shared and two private records
    pressure.scope_shared_bytes = 200;
    pressure.scope_cpu_limit = 100;
    check(pressure.scope_private_bytes() == 200, "private bytes include shared payloads");
    HistoricalLruPolicy policy(8);
    HistoricalDecisionBatch out(8);
    auto result = policy.step(s, pressure, {8, 8}, out);
    check(result.emitted == 1 && out.values()[0].handle.slot == 2,
          "a byte budget retired a shared record or more than its private deficit");
    out.clear_after_ack();
    pressure.scope_cpu_limit = UINT64_MAX;
    pressure.requested_ready_slots = 8;   // one slot short of the request
    pressure.ready_slots = 7;
    HistoricalLruPolicy slots_policy(8);
    result = slots_policy.step(s, pressure, {8, 8}, out);
    check(result.emitted == 1 && out.values()[0].handle.slot == 0,
          "a slot deficit could not retire the oldest (shared) record");
    out.clear_after_ack();
    pressure.requested_ready_slots = 0;
    pressure.scope_cpu_bytes = 200;
    pressure.scope_shared_bytes = 200;    // nothing private: no byte deficit at all
    pressure.scope_cpu_limit = 0;
    check(!policy.wants_snapshot(pressure), "shared bytes created a private byte deficit");
}

static void refill_reuses_one_scan_and_observes_current_pressure() {
    HistoricalSnapshot s{7, 1, {}};
    for (uint32_t i = 0; i < 1024; ++i) s.blocks.push_back(block(i, i, i));
    HistoricalPressure pressure{0, 128, 256, 0};
    HistoricalLruPolicy policy(1024);
    HistoricalDecisionBatch out(64);
    size_t examined = 0, emitted = 0;
    for (size_t tick = 0; tick < 100; ++tick) {
        auto result = policy.step(s, pressure, {256, 64}, out);
        examined += result.examined;
        for (const auto &intent : out.values()) {
            check(intent.handle.slot == emitted, "LRU batch order changed");
            ++emitted;
        }
        pressure.pending_slots += static_cast<uint32_t>(out.values().size());
        out.clear_after_ack();
        if (result.status == HistoricalPolicyStatus::Complete) break;
    }
    check(examined == 1024 && emitted == 256, "refill rescanned the repository per batch");

    HistoricalSnapshot small{9, 1, {block(0, 1, 1), block(1, 2, 2), block(2, 3, 3)}};
    HistoricalLruPolicy next(8);
    HistoricalDecisionBatch decisions(2);
    HistoricalPressure fresh{0, 1, 2, 0};
    next.step(small, fresh, {1, 2}, decisions);
    fresh.ready_slots = 2; // An unrelated request returned private reservations.
    check(next.step(small, fresh, {8, 2}, decisions).emitted == 0,
          "stale pressure caused deletion after capacity was returned");
}

static void eligibility_is_a_policy_choice_and_snapshot_is_stable() {
    HistoricalSnapshot s{7, 1, {block(0, 1, 1), block(1, 2, 2), block(2, 3, 3)}};
    s.blocks[0].mandatory = true;
    s.blocks[1].sealed = false;
    HistoricalPressure p{0, 1, 2, 0};
    HistoricalLruPolicy keep(8), allow(8, false);
    HistoricalDecisionBatch out(8);
    check(keep.step(s, p, {8, 8}, out).emitted == 1 && out.values()[0].handle.slot == 2,
          "default policy failed its explicit sealed/mandatory rules");
    out.clear_after_ack();
    check(allow.step(s, p, {8, 8}, out).emitted == 2 && out.values()[0].handle.slot == 0,
          "mandatory exclusion leaked into the retirement mechanism");
    out.clear_after_ack();
    HistoricalLruPolicy sliced(8);
    sliced.step(s, p, {1, 1}, out);
    ++s.round;
    bool rejected = false;
    try { sliced.step(s, p, {1, 1}, out); }
    catch (const std::logic_error &) { rejected = true; }
    check(rejected, "policy silently reset a partially scanned snapshot");
}

static void factory_validates_parameters_and_preserves_policy_behavior() {
    HistoricalSnapshot s{7, 1, {block(0, 1, 1), block(1, 2, 2)}};
    s.blocks[0].mandatory = true;
    const HistoricalPressure pressure{0, 1, 2, 0};
    HistoricalDecisionBatch out(8);
    auto keep = make_historical_kv_policy("lru", {}, 8);
    keep->step(s, pressure, {8, 8}, out);
    check(out.values().size() == 1 && out.values()[0].handle.slot == 1,
          "factory default changed mandatory protection");
    out.clear_after_ack();
    auto allow = make_historical_kv_policy("lru", {{"keep_mandatory", false}}, 8);
    allow->step(s, pressure, {8, 8}, out);
    check(out.values().size() == 2 && out.values()[0].handle.slot == 0,
          "factory ignored explicit LRU parameters");
    out.clear_after_ack();
    allow->reset();
    ++s.incarnation;
    ++s.round;
    allow->step(s, pressure, {8, 8}, out);
    check(out.values().size() == 2, "factory policy reset lost configuration or round state");
    out.clear_after_ack();
    auto legacy = make_historical_kv_policy("lru", {}, 8, false);
    legacy->step(s, pressure, {8, 8}, out);
    check(out.values().size() == 2, "legacy native keep_mandatory fallback was lost");
    out.clear_after_ack();
    auto explicit_keep = make_historical_kv_policy("lru", {{"keep_mandatory", true}}, 8, false);
    explicit_keep->step(s, pressure, {8, 8}, out);
    check(out.values().size() == 1, "explicit parameters did not override the legacy fallback");

    const std::vector<std::pair<std::string, HistoricalPolicyParameters>> invalid = {
        {"fifo", {}}, {"", {}}, {"lru", {{"typo", true}}},
        {"lru", {{"keep_mandatory", int64_t(1)}}},
        {"lru", {{"keep_mandatory", 1.0}}},
        {"lru", {{"keep_mandatory", std::string("true")}}}
    };
    for (const auto &[name, params] : invalid) {
        bool rejected = false;
        try { make_historical_kv_policy(name, params, 8); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "factory accepted an unknown policy or invalid parameter");
    }
}


static HistoricalBlockMeta thinking(uint32_t slot, uint32_t turn, bool pure = true) {
    auto b = block(slot, slot, slot);
    b.thinking_complete = true;
    b.thinking_only = pure;
    b.thinking_turn = turn;
    return b;
}

static void thinking_age_retires_old_pure_thinking_without_pressure() {
    const HistoricalPolicyParameters params{{"thinking_age_steps", int64_t(2)}};
    validate_historical_kv_policy("lru", params);
    validate_historical_kv_policy("redundancy", {{"thinking_age_steps", int64_t(10)}, {"thinking_only", true}});
    auto policy = make_historical_kv_policy("lru", params, 16);
    HistoricalSnapshot snapshot;
    snapshot.incarnation = 1;
    snapshot.round = 1;
    snapshot.blocks = {thinking(0, 1), thinking(1, 2), thinking(2, 1, false), thinking(3, 0),
                       block(4, 4, 4), thinking(5, 1)};
    snapshot.blocks[5].mandatory = true;    // current window, recent range or a hold
    HistoricalPressure calm;
    calm.ready_slots = 10; calm.low_slots = 2; calm.high_slots = 4;
    calm.thinking_turns = 2;
    check(!policy->wants_snapshot(calm), "age rule armed before a turn N steps newer completed");
    calm.thinking_turns = 3;
    check(policy->wants_snapshot(calm), "a newly completed turn did not start an age pass");
    HistoricalDecisionBatch batch(8);
    const auto result = policy->step(snapshot, calm, {16, 8}, batch);
    check(result.emitted == 1 && batch.values().size() == 1 && batch.values()[0].handle == snapshot.blocks[0].handle &&
          result.thinking_age_protected == -1 &&
          batch.values()[0].reason == HistoricalDeleteReason::ThinkingAge,
          "age pass must emit only old, pure, unprotected thinking with a known turn");
    check(policy->permits_pressure(calm), "an age batch was withdrawn for lack of capacity pressure");
    check(policy->permits_intent(snapshot.blocks[0], batch.values()[0]), "eligible age intent was refused");
    calm.thinking_turns = 2;    // a rewind lowered the completed turn count
    check(policy->permits_pressure(calm) && !policy->permits_intent(snapshot.blocks[0], batch.values()[0]),
          "age intent survived a rewind below its threshold");
    calm.thinking_turns = 3;
    policy->permits_pressure(calm);
    batch.clear_after_ack();
    const auto done = policy->step(snapshot, calm, {16, 8}, batch);
    check(done.status == HistoricalPolicyStatus::Complete && batch.empty() && done.thinking_age_protected == 1,
          "calm round did not end after the age pass, or missed the protected record");
    check(!policy->wants_snapshot(calm), "unchanged turn count rescanned without pressure");
    calm.thinking_pure = 63;
    check(!policy->wants_snapshot(calm), "a few new thinking records started a pass");
    calm.thinking_pure = 64;
    check(policy->wants_snapshot(calm), "64 newly classified thinking records did not re-arm the age pass");

    // Under capacity pressure the wrapped policy still runs after the age pass.
    snapshot.round = 2;
    HistoricalPressure tight = calm;
    tight.ready_slots = 1;
    tight.thinking_turns = 4;
    check(policy->wants_snapshot(tight), "pressure round was not requested");
    auto first = policy->step(snapshot, tight, {16, 8}, batch);
    check(first.emitted == 2 && batch.values()[0].reason == HistoricalDeleteReason::ThinkingAge &&
          batch.values()[1].reason == HistoricalDeleteReason::ThinkingAge,
          "turns 1 and 2 were not both retired by age before the pressure policy");
    batch.clear_after_ack();
    auto second = policy->step(snapshot, tight, {16, 8}, batch);
    check(second.emitted && batch.values()[0].reason == HistoricalDeleteReason::Lru,
          "wrapped LRU did not run after the age pass");
    for (const auto &intent : batch.values())
        check(intent.handle != snapshot.blocks[0].handle && intent.handle != snapshot.blocks[1].handle,
              "wrapped LRU re-proposed records the age pass already took this round");
    check(policy->permits_pressure(tight), "wrapped policy's pressure recheck changed");

    bool rejected = false;
    try { make_historical_kv_policy("lru", {{"thinking_age_steps", int64_t(-1)}}, 16); }
    catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "negative thinking age was accepted");
    auto plain = make_historical_kv_policy("lru", {{"thinking_age_steps", int64_t(0)}}, 16);
    check(dynamic_cast<HistoricalLruPolicy *>(plain.get()) != nullptr, "zero thinking age wrapped the policy");
}

int main() {
    try {
        irreversible_retirement();
        legacy_retirement_can_be_finalized();
        stale_intents_racing_with_slot_reuse_do_not_delete_new_generations();
        policy_only_emits_intents();
        byte_pressure_is_independent_of_slots();
        shared_records_count_only_against_slots();
        refill_reuses_one_scan_and_observes_current_pressure();
        eligibility_is_a_policy_choice_and_snapshot_is_stable();
        factory_validates_parameters_and_preserves_policy_behavior();
        thinking_age_retires_old_pure_thinking_without_pressure();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Historical KV irreversible retirement and LRU policy tests passed\n";
}
