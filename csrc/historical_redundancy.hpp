// Host-only bounded Mean-K/Query observations for opt-in redundancy GC.
#pragma once

#include "historical_kv_policy.hpp"

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace vllm_kvmem {

class HistoricalFeatureSource {
public:
    using Handle = qw3::KvMemBlockHandle;
    struct Key {
        Handle handle{};
        uint64_t version = 0;
        std::vector<float> values;
        std::vector<double> inverse_norms;
        std::shared_ptr<void> charge;
    };
    struct QueryBank {
        size_t width = 0, segments = 0;
        std::vector<float> values; // [segment,L,Hq,D]
        std::vector<double> sums; // private until complete
        std::shared_ptr<void> charge;
    };
    struct Snapshot {
        uint64_t epoch = 0;
        std::array<size_t, 4> geometry{};
        std::shared_ptr<const QueryBank> question, observations;
        bool valid() const { return epoch && question && question->segments; }
        size_t segments() const {
            return question ? question->segments + (observations ? observations->segments : 0) : 0;
        }
        const float *segment(size_t i) const;
    };
    struct CommitGuard {
        bool acquired = false, valid = false;
        std::unique_lock<std::mutex> lock;
    };

    HistoricalFeatureSource(size_t capacity, size_t feature_cache_bytes, size_t query_cache_bytes);
    ~HistoricalFeatureSource();
    void configure(size_t layers, size_t kv_heads, size_t q_heads, size_t dim);
    std::array<size_t, 4> geometry() const;
    size_t key_elements() const;
    size_t query_elements_per_token() const;
    bool publish_key(Handle, const float *, size_t count);
    uint64_t begin_question(size_t expected_tokens);
    bool append_question(uint64_t epoch, size_t offset, const float *, size_t token_count);
    bool observe(uint64_t epoch, const float *, size_t token_count);
    void invalidate_question();
    void erase_key(Handle);
    void reset();
    uint64_t key_version(Handle) const;
    std::shared_ptr<const Key> key(Handle) const;
    Snapshot snapshot() const;
    CommitGuard try_commit_guard(const HistoricalDeleteIntent &) const;
    std::map<std::string, uint64_t> stats() const;
private:
    struct State;
    std::unique_ptr<State> state_;
};

struct HistoricalRedundancyConfig {
    bool keep_mandatory = true;
    double importance_weight = .1, representative_similarity = .9;
    size_t reference_blocks = 32, candidate_blocks = 128, pair_comparisons = 512;
    size_t query_segments_per_step = 8;
    size_t feature_cache_bytes = 134217728, query_cache_bytes = 8388608;
    bool allow_lossy_fallback = false;
};

class HistoricalRedundancyPolicy final : public HistoricalKvPolicy {
public:
    HistoricalRedundancyPolicy(size_t capacity, HistoricalRedundancyConfig);
    ~HistoricalRedundancyPolicy() override;
    void bind_features(std::shared_ptr<const HistoricalFeatureSource>) override;
    void reset() override;
    bool wants_snapshot(const HistoricalPressure &) override;
    bool permits(const HistoricalBlockMeta &) const override;
    HistoricalPolicyResult step(const HistoricalSnapshot &, const HistoricalPressure &,
                               HistoricalWorkBudget, HistoricalDecisionBatch &) override;
    void on_result(const HistoricalDeleteIntent &, HistoricalCommitCode) noexcept override;
private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace vllm_kvmem
