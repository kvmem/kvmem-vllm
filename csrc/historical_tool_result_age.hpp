#pragma once

#include "historical_kv_policy.hpp"

namespace vllm_kvmem {

// The engine publishes expiration from accepted, complete tool replies. The
// worker scans on those changes and on backing/protection changes, regardless
// of capacity. There is no fallback to eviction of other message roles.
class HistoricalToolResultAgePolicy final : public HistoricalKvPolicy {
public:
    void reset() override {
        completed_revision_ = round_ = 0;
        cursor_ = 0;
    }

    bool wants_snapshot(const HistoricalPressure &pressure) override {
        return pressure.tool_result_revision && pressure.tool_result_revision != completed_revision_;
    }

    bool permits(const HistoricalBlockMeta &block) const override {
        return block.live && block.sealed && !block.mandatory && block.expired_tool_result;
    }

    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot, const HistoricalPressure &,
                                HistoricalWorkBudget budget, HistoricalDecisionBatch &out) override {
        if (!budget.scan_blocks || !budget.delete_blocks)
            throw std::invalid_argument("Historical work budgets must be positive");
        if (!out.empty()) return {HistoricalPolicyStatus::Backpressure, 0, 0};
        if (round_ != snapshot.round) {
            round_ = snapshot.round;
            cursor_ = 0;
        }
        size_t examined = 0, emitted = 0;
        while (cursor_ < snapshot.blocks.size() && examined < budget.scan_blocks &&
               emitted < budget.delete_blocks && out.remaining()) {
            const auto &block = snapshot.blocks[cursor_++];
            ++examined;
            if (!permits(block)) continue;
            out.push({block.handle, snapshot.incarnation, snapshot.round, HistoricalDeleteReason::ToolResultAge});
            ++emitted;
        }
        if (cursor_ < snapshot.blocks.size())
            return {HistoricalPolicyStatus::Yielded, examined, emitted};
        completed_revision_ = snapshot.tool_result_revision;
        return {HistoricalPolicyStatus::Complete, examined, emitted};
    }

private:
    uint64_t completed_revision_ = 0, round_ = 0;
    size_t cursor_ = 0;
};

} // namespace vllm_kvmem
