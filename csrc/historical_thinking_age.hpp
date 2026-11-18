// Proactive age rule for historical thinking (opt-in, policy_params.thinking_age_steps).
#pragma once

#include "historical_kv_policy.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vllm_kvmem {

// Retires a record of pure thinking once the assistant turn `steps` turns
// newer than the record's own turn has completed: by then the step's plan has
// been carried out, and its tool calls and results stay in history. The rule
// needs no capacity pressure. It wraps the configured pressure policy, which
// keeps its own behavior and runs after the age pass of each round.
//
// Mechanism checks are unchanged: the manager never retires the current
// window, the recent range or a held record, so such a record is retried once
// another assistant turn completes. Boundary records that mix thinking with
// other tokens are never pure and are kept.
class HistoricalThinkingAgePolicy final : public HistoricalKvPolicy {
public:
    HistoricalThinkingAgePolicy(std::unique_ptr<HistoricalKvPolicy> inner, uint32_t steps)
        : inner_(std::move(inner)), steps_(steps) {
        if (!inner_ || !steps_)
            throw std::invalid_argument("Thinking age GC requires a pressure policy and a positive step count");
    }

    void reset() override {
        inner_->reset();
        initialized_ = age_active_ = inner_round_ = age_batch_ = false;
        cursor_ = protected_ = 0;
        taken_.clear();
        filtered_.blocks.clear();
        inner_snapshot_ = nullptr;
        incarnation_ = round_ = round_turns_ = turns_ = scanned_turns_ = round_pure_ = scanned_pure_ = 0;
    }
    void bind_features(std::shared_ptr<const HistoricalFeatureSource> features) override {
        inner_->bind_features(std::move(features));
    }
    void on_result(const HistoricalDeleteIntent &intent, HistoricalCommitCode code) noexcept override {
        if (intent.reason != HistoricalDeleteReason::ThinkingAge) inner_->on_result(intent, code);
    }
    bool wants_snapshot(const HistoricalPressure &pressure) override {
        turns_ = pressure.thinking_turns;
        inner_round_ = inner_->wants_snapshot(pressure);
        // A pass per newly completed turn; it also retries records that the
        // current window or recent range protected last time. Records keep
        // arriving after the turn count (a long prefill, asynchronous
        // archiving), so newly classified thinking re-arms it in batches.
        return inner_round_ || (armed(turns_) && (turns_ != scanned_turns_ ||
            pressure.thinking_pure >= scanned_pure_ + rescan_records));
    }
    bool permits_pressure(const HistoricalPressure &pressure) override {
        turns_ = pressure.thinking_turns;
        // An age batch is not a capacity decision; a withdrawn reserve keeps it.
        return age_batch_ || inner_->permits_pressure(pressure);
    }
    bool permits(const HistoricalBlockMeta &block) const override {
        return eligible(block, turns_) || inner_->permits(block);
    }
    bool permits_intent(const HistoricalBlockMeta &block, const HistoricalDeleteIntent &intent) const override {
        // Rechecked with the latest turn count: a rewind can lower it.
        if (intent.reason == HistoricalDeleteReason::ThinkingAge) return eligible(block, turns_);
        return inner_->permits_intent(block, intent);
    }

    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot, const HistoricalPressure &pressure,
                                HistoricalWorkBudget budget, HistoricalDecisionBatch &out) override {
        if (!budget.scan_blocks || !budget.delete_blocks)
            throw std::invalid_argument("Historical work budgets must be positive");
        if (!out.empty()) return {HistoricalPolicyStatus::Backpressure, 0, 0};
        if (!initialized_ || incarnation_ != snapshot.incarnation || round_ != snapshot.round) {
            initialized_ = true;
            incarnation_ = snapshot.incarnation;
            round_ = snapshot.round;
            round_turns_ = pressure.thinking_turns;
            round_pure_ = pressure.thinking_pure;
            age_active_ = armed(round_turns_);
            cursor_ = protected_ = 0;
            taken_.clear();
            inner_snapshot_ = nullptr;
        }
        age_batch_ = false;
        size_t examined = 0, emitted = 0;
        while (age_active_ && cursor_ < snapshot.blocks.size() && examined < budget.scan_blocks &&
               emitted < budget.delete_blocks && out.remaining()) {
            const auto &block = snapshot.blocks[cursor_++];
            ++examined;
            if (!old_thinking(block, round_turns_)) continue;
            if (block.mandatory) {
                ++protected_;    // Retried once another turn completes.
                continue;
            }
            if (!eligible(block, round_turns_)) continue;
            out.push({block.handle, snapshot.incarnation, snapshot.round, HistoricalDeleteReason::ThinkingAge});
            taken_.push_back(block.handle);
            ++emitted;
        }
        if (emitted) {
            age_batch_ = true;
            return {HistoricalPolicyStatus::Yielded, examined, emitted};
        }
        if (age_active_ && cursor_ < snapshot.blocks.size())
            return {HistoricalPolicyStatus::Yielded, examined, 0};
        int64_t reported = -1;
        if (age_active_) {
            age_active_ = false;
            scanned_turns_ = round_turns_;
            scanned_pure_ = round_pure_;
            reported = static_cast<int64_t>(protected_);
        }
        if (!inner_round_) {
            HistoricalPolicyResult done{HistoricalPolicyStatus::Complete, examined, 0, examined != 0};
            done.thinking_age_protected = reported;
            return done;
        }
        if (!inner_snapshot_) {
            // The wrapped policy sees this round without the records the age
            // pass already proposed, so it never spends its deficit on them.
            inner_snapshot_ = &snapshot;
            if (!taken_.empty()) {
                filtered_.incarnation = snapshot.incarnation;
                filtered_.round = snapshot.round;
                filtered_.blocks.clear();
                std::sort(taken_.begin(), taken_.end(), handle_less);
                for (const auto &block : snapshot.blocks)
                    if (!std::binary_search(taken_.begin(), taken_.end(), block.handle, handle_less))
                        filtered_.blocks.push_back(block);
                inner_snapshot_ = &filtered_;
            }
        }
        auto result = inner_->step(*inner_snapshot_, pressure, budget, out);
        result.examined += examined;
        result.progressed = result.progressed || examined != 0;
        result.thinking_age_protected = reported;
        return result;
    }

private:
    bool armed(uint64_t turns) const { return turns > steps_; }
    bool old_thinking(const HistoricalBlockMeta &block, uint64_t turns) const {
        return block.thinking_only && block.thinking_turn && uint64_t(block.thinking_turn) + steps_ <= turns;
    }
    bool eligible(const HistoricalBlockMeta &block, uint64_t turns) const {
        return block.live && block.sealed && block.handle.generation && !block.mandatory &&
               old_thinking(block, turns);
    }
    static bool handle_less(const qw3::KvMemBlockHandle &a, const qw3::KvMemBlockHandle &b) {
        return std::make_pair(a.slot, a.generation) < std::make_pair(b.slot, b.generation);
    }

    std::unique_ptr<HistoricalKvPolicy> inner_;
    const uint32_t steps_;
    size_t cursor_ = 0, protected_ = 0;
    std::vector<qw3::KvMemBlockHandle> taken_;
    HistoricalSnapshot filtered_;
    const HistoricalSnapshot *inner_snapshot_ = nullptr;
    // 64 records = 2,048 tokens of new thinking at 32-token records.
    static constexpr uint64_t rescan_records = 64;
    uint64_t incarnation_ = 0, round_ = 0, round_turns_ = 0, turns_ = 0, scanned_turns_ = 0;
    uint64_t round_pure_ = 0, scanned_pure_ = 0;
    bool initialized_ = false, age_active_ = false, inner_round_ = false, age_batch_ = false;
};

} // namespace vllm_kvmem
