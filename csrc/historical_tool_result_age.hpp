#pragma once

#include "historical_kv_policy.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vllm_kvmem {

// The engine publishes expiration from accepted, complete tool replies. The
// worker scans on those changes and on backing/protection changes, regardless
// of capacity. Without an inner policy there is no fallback to eviction of
// other message roles: a capacity shortfall waits at admission.
//
// With an inner pressure policy (policy_params.lru_fallback), the age pass
// runs first in each round, then the inner policy handles capacity pressure
// on the records the age pass did not propose (as the thinking age wrapper).
class HistoricalToolResultAgePolicy final : public HistoricalKvPolicy {
public:
    HistoricalToolResultAgePolicy() = default;
    explicit HistoricalToolResultAgePolicy(std::unique_ptr<HistoricalKvPolicy> inner) : inner_(std::move(inner)) {
        if (!inner_) throw std::invalid_argument("Tool result age fallback requires a pressure policy");
    }

    void reset() override {
        completed_revision_ = round_ = 0;
        cursor_ = 0;
        if (!inner_) return;
        inner_->reset();
        initialized_ = age_active_ = inner_round_ = age_batch_ = false;
        incarnation_ = round_revision_ = 0;
        taken_.clear();
        filtered_.blocks.clear();
        inner_snapshot_ = nullptr;
    }
    void bind_features(std::shared_ptr<const HistoricalFeatureSource> features) override {
        if (inner_) inner_->bind_features(std::move(features));
    }
    void on_result(const HistoricalDeleteIntent &intent, HistoricalCommitCode code) noexcept override {
        if (inner_ && intent.reason != HistoricalDeleteReason::ToolResultAge) inner_->on_result(intent, code);
    }

    bool wants_snapshot(const HistoricalPressure &pressure) override {
        const bool age = pressure.tool_result_revision && pressure.tool_result_revision != completed_revision_;
        if (!inner_) return age;
        inner_round_ = inner_->wants_snapshot(pressure);
        return age || inner_round_;
    }
    bool permits_pressure(const HistoricalPressure &pressure) override {
        // An age batch is not a capacity decision; a withdrawn reserve keeps it.
        return !inner_ || age_batch_ || inner_->permits_pressure(pressure);
    }
    bool permits(const HistoricalBlockMeta &block) const override {
        return expired(block) || (inner_ && inner_->permits(block));
    }
    bool permits_intent(const HistoricalBlockMeta &block, const HistoricalDeleteIntent &intent) const override {
        if (!inner_ || intent.reason == HistoricalDeleteReason::ToolResultAge) return expired(block);
        return inner_->permits_intent(block, intent);
    }

    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot, const HistoricalPressure &pressure,
                                HistoricalWorkBudget budget, HistoricalDecisionBatch &out) override {
        if (!budget.scan_blocks || !budget.delete_blocks)
            throw std::invalid_argument("Historical work budgets must be positive");
        if (!out.empty()) return {HistoricalPolicyStatus::Backpressure, 0, 0};
        if (inner_) return step_with_fallback(snapshot, pressure, budget, out);
        if (round_ != snapshot.round) {
            round_ = snapshot.round;
            cursor_ = 0;
        }
        size_t examined = 0, emitted = 0;
        while (cursor_ < snapshot.blocks.size() && examined < budget.scan_blocks &&
               emitted < budget.delete_blocks && out.remaining()) {
            const auto &block = snapshot.blocks[cursor_++];
            ++examined;
            if (!expired(block)) continue;
            out.push({block.handle, snapshot.incarnation, snapshot.round, HistoricalDeleteReason::ToolResultAge});
            ++emitted;
        }
        if (cursor_ < snapshot.blocks.size())
            return {HistoricalPolicyStatus::Yielded, examined, emitted};
        completed_revision_ = snapshot.tool_result_revision;
        return {HistoricalPolicyStatus::Complete, examined, emitted};
    }

private:
    static bool expired(const HistoricalBlockMeta &block) {
        return block.live && block.sealed && !block.mandatory && block.expired_tool_result;
    }
    static bool handle_less(const qw3::KvMemBlockHandle &a, const qw3::KvMemBlockHandle &b) {
        return std::make_pair(a.slot, a.generation) < std::make_pair(b.slot, b.generation);
    }

    HistoricalPolicyResult step_with_fallback(const HistoricalSnapshot &snapshot, const HistoricalPressure &pressure,
                                              HistoricalWorkBudget budget, HistoricalDecisionBatch &out) {
        if (!initialized_ || incarnation_ != snapshot.incarnation || round_ != snapshot.round) {
            initialized_ = true;
            incarnation_ = snapshot.incarnation;
            round_ = snapshot.round;
            round_revision_ = snapshot.tool_result_revision;
            age_active_ = round_revision_ && round_revision_ != completed_revision_;
            cursor_ = 0;
            taken_.clear();
            inner_snapshot_ = nullptr;
        }
        age_batch_ = false;
        size_t examined = 0, emitted = 0;
        while (age_active_ && cursor_ < snapshot.blocks.size() && examined < budget.scan_blocks &&
               emitted < budget.delete_blocks && out.remaining()) {
            const auto &block = snapshot.blocks[cursor_++];
            ++examined;
            if (!expired(block)) continue;
            out.push({block.handle, snapshot.incarnation, snapshot.round, HistoricalDeleteReason::ToolResultAge});
            taken_.push_back(block.handle);
            ++emitted;
        }
        if (emitted) {
            age_batch_ = true;
            return {HistoricalPolicyStatus::Yielded, examined, emitted};
        }
        if (age_active_ && cursor_ < snapshot.blocks.size())
            return {HistoricalPolicyStatus::Yielded, examined, 0};
        if (age_active_) {
            age_active_ = false;
            completed_revision_ = round_revision_;
        }
        if (!inner_round_) return {HistoricalPolicyStatus::Complete, examined, 0, examined != 0};
        if (!inner_snapshot_) {
            // The pressure policy sees this round without the records the age
            // pass already proposed, so it never spends its deficit on them.
            inner_snapshot_ = &snapshot;
            if (!taken_.empty()) {
                filtered_.incarnation = snapshot.incarnation;
                filtered_.round = snapshot.round;
                filtered_.tool_result_revision = snapshot.tool_result_revision;
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
        return result;
    }

    std::unique_ptr<HistoricalKvPolicy> inner_;
    uint64_t completed_revision_ = 0, round_ = 0;
    size_t cursor_ = 0;
    // Fallback mode only.
    std::vector<qw3::KvMemBlockHandle> taken_;
    HistoricalSnapshot filtered_;
    const HistoricalSnapshot *inner_snapshot_ = nullptr;
    uint64_t incarnation_ = 0, round_revision_ = 0;
    bool initialized_ = false, age_active_ = false, inner_round_ = false, age_batch_ = false;
};

} // namespace vllm_kvmem
