// Construction and configuration only; never called by a decode/policy step.
#pragma once

#include "historical_kv_policy.hpp"
#include "historical_redundancy.hpp"
#include "historical_thinking_age.hpp"
#include "historical_tool_result_age.hpp"

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <cmath>
#include <utility>

namespace vllm_kvmem {

using HistoricalPolicyValue = std::variant<bool, int64_t, double, std::string>;
using HistoricalPolicyParameters = std::map<std::string, HistoricalPolicyValue>;

namespace historical_policy_detail {
inline bool lru_keep_mandatory(const HistoricalPolicyParameters &params, bool fallback) {
    for (const auto &[key, value] : params) {
        if (key != "keep_mandatory")
            throw std::invalid_argument("Unknown historical_gc.policy_params.lru parameter: " + key);
        const auto *keep = std::get_if<bool>(&value);
        if (!keep)
            throw std::invalid_argument("historical_gc.policy_params.keep_mandatory must be boolean");
        fallback = *keep;
    }
    return fallback;
}
// thinking_age_steps applies to every policy: it wraps the configured one
// with the proactive age rule. Returns the step count (0 = off) and the
// remaining parameters for that policy.
inline std::pair<uint32_t, HistoricalPolicyParameters> split_thinking_age(
        const HistoricalPolicyParameters &params) {
    auto rest = params;
    const auto it = rest.find("thinking_age_steps");
    if (it == rest.end()) return {0u, rest};
    const auto *steps = std::get_if<int64_t>(&it->second);
    if (!steps || *steps < 0 || *steps > 1000000)
        throw std::invalid_argument("historical_gc.policy_params.thinking_age_steps must be an integer in [0, 1000000]");
    const auto value = static_cast<uint32_t>(*steps);
    rest.erase(it);
    return {value, rest};
}
[[noreturn]] inline void unknown(const std::string &name) {
    throw std::invalid_argument("Unknown historical_gc.policy: " + name + " (supported: lru, redundancy, tool_result_age)");
}
inline void validate_tool_result_age(const HistoricalPolicyParameters &params) {
    for (const auto &[key, value] : params) {
        if (key == "tool_result_age_steps") {
            const auto *steps = std::get_if<int64_t>(&value);
            if (!steps || *steps < 1 || *steps > 1000000)
                throw std::invalid_argument("historical_gc.policy_params.tool_result_age_steps must be an integer in [1, 1000000]");
        } else if (key == "keep_mandatory") {
            const auto *keep = std::get_if<bool>(&value);
            if (!keep || !*keep)
                throw std::invalid_argument("historical_gc tool_result_age requires keep_mandatory=true");
        } else if (key == "lru_fallback") {
            if (!std::get_if<bool>(&value))
                throw std::invalid_argument("historical_gc.policy_params.lru_fallback must be boolean");
        } else {
            throw std::invalid_argument("Unknown historical_gc.policy_params.tool_result_age parameter: " + key);
        }
    }
}
// tool_result_age with lru_fallback: LRU (keep_mandatory) handles capacity
// pressure once the age pass has proposed its expired tool pages.
inline bool tool_result_age_lru_fallback(const HistoricalPolicyParameters &params) {
    const auto it = params.find("lru_fallback");
    if (it == params.end()) return false;
    const auto *value = std::get_if<bool>(&it->second);
    return value && *value;
}
inline HistoricalRedundancyConfig redundancy_config(const HistoricalPolicyParameters &params) {
    HistoricalRedundancyConfig config;
    for (const auto &[key,value]:params) {
        auto boolean=[&]() {
            const auto *v=std::get_if<bool>(&value);
            if(!v) throw std::invalid_argument("historical_gc.policy_params."+key+" must be boolean");
            return *v;
        };
        auto integer=[&](uint64_t limit) {
            const auto *v=std::get_if<int64_t>(&value);
            if(!v || *v<=0 || uint64_t(*v)>limit)
                throw std::invalid_argument("Invalid historical_gc.policy_params."+key+" integer bound");
            return size_t(*v);
        };
        auto number=[&](double minimum,double maximum) {
            double v;
            if(const auto *p=std::get_if<double>(&value)) v=*p;
            else if(const auto *p=std::get_if<int64_t>(&value)) v=double(*p);
            else throw std::invalid_argument("historical_gc.policy_params."+key+" must be numeric, not boolean");
            if(!std::isfinite(v) || v<minimum || v>maximum)
                throw std::invalid_argument("Invalid historical_gc.policy_params."+key+" numeric bound");
            return v;
        };
        if(key=="keep_mandatory") config.keep_mandatory=boolean();
        else if(key=="allow_lossy_fallback") config.allow_lossy_fallback=boolean();
        else if(key=="thinking_only") config.thinking_only=boolean();
        else if(key=="pool_pairwise") config.pool_pairwise=boolean();
        else if(key=="all_history_references") config.all_history_references=boolean();
        else if(key=="refresh_before_rank") config.refresh_before_rank=boolean();
        else if(key=="epoch_peak") config.epoch_peak=boolean();
        else if(key=="lru_fallback") config.lru_fallback=boolean();
        else if(key=="importance_weight") config.importance_weight=number(0,1);
        else if(key=="representative_similarity") config.representative_similarity=number(-1,1);
        else if(key=="reference_blocks") config.reference_blocks=integer(65536);
        else if(key=="candidate_blocks") config.candidate_blocks=integer(65536);
        else if(key=="pair_comparisons") config.pair_comparisons=integer(65536);
        else if(key=="query_segments_per_step") config.query_segments_per_step=integer(65536);
        else if(key=="feature_cache_bytes") config.feature_cache_bytes=integer(uint64_t(1)<<40);
        else if(key=="query_cache_bytes") config.query_cache_bytes=integer(uint64_t(1)<<40);
        else if(key=="query_capture_bytes") config.query_capture_bytes=integer(uint64_t(1)<<30);
        else throw std::invalid_argument("Unknown historical_gc.policy_params.redundancy parameter: "+key);
    }
    if(!config.keep_mandatory) throw std::invalid_argument("Redundancy GC requires keep_mandatory=true");
    if(config.lru_fallback && config.allow_lossy_fallback)
        throw std::invalid_argument("Choose lru_fallback or allow_lossy_fallback, not both");
    return config;
}
} // namespace historical_policy_detail

inline void validate_historical_kv_policy(const std::string &name,
                                          const HistoricalPolicyParameters &all_params) {
    if (name == "tool_result_age") {
        historical_policy_detail::validate_tool_result_age(all_params);
        return;
    }
    const auto params = historical_policy_detail::split_thinking_age(all_params).second;
    if (name == "lru") {
        historical_policy_detail::lru_keep_mandatory(params, true);
        return;
    }
    if (name == "redundancy") {
        historical_policy_detail::redundancy_config(params);
        return;
    }
    historical_policy_detail::unknown(name);
}

namespace historical_policy_detail {
// Add each built-in policy's validation/construction here. Policies only emit
// intents; all implementations share the manager's existing release protocol.
// The fallback preserves the earlier native Config.keep_mandatory API; an
// explicit policy parameter takes precedence.
inline std::unique_ptr<HistoricalKvPolicy> make_pressure_policy(
        const std::string &name, const HistoricalPolicyParameters &params,
        size_t capacity, bool keep_mandatory) {
    if (name == "lru")
        return std::make_unique<HistoricalLruPolicy>(capacity,
            historical_policy_detail::lru_keep_mandatory(params, keep_mandatory));
    if (name == "redundancy") {
        auto config=historical_policy_detail::redundancy_config(params);
        if(!keep_mandatory && !params.count("keep_mandatory"))
            throw std::invalid_argument("Redundancy GC requires keep_mandatory=true");
        return std::make_unique<HistoricalRedundancyPolicy>(capacity,config);
    }
    historical_policy_detail::unknown(name);
}
} // namespace historical_policy_detail

inline std::unique_ptr<HistoricalKvPolicy> make_historical_kv_policy(
        const std::string &name, const HistoricalPolicyParameters &params,
        size_t capacity, bool keep_mandatory = true) {
    if (name == "tool_result_age") {
        historical_policy_detail::validate_tool_result_age(params);
        if (historical_policy_detail::tool_result_age_lru_fallback(params))
            return std::make_unique<HistoricalToolResultAgePolicy>(
                std::make_unique<HistoricalLruPolicy>(capacity, true));
        return std::make_unique<HistoricalToolResultAgePolicy>();
    }
    const auto [steps, rest] = historical_policy_detail::split_thinking_age(params);
    auto policy = historical_policy_detail::make_pressure_policy(name, rest, capacity, keep_mandatory);
    if (!steps) return policy;
    return std::make_unique<HistoricalThinkingAgePolicy>(std::move(policy), steps);
}

inline std::shared_ptr<HistoricalFeatureSource> make_historical_feature_source(
        const std::string &name,const HistoricalPolicyParameters &all_params,size_t capacity) {
    if (name == "tool_result_age") {
        historical_policy_detail::validate_tool_result_age(all_params);
        return {};
    }
    const auto params = historical_policy_detail::split_thinking_age(all_params).second;
    if(name=="lru") {historical_policy_detail::lru_keep_mandatory(params,true);return {};}
    if(name=="redundancy") {
        const auto config=historical_policy_detail::redundancy_config(params);
        return std::make_shared<HistoricalFeatureSource>(capacity,config.feature_cache_bytes,config.query_cache_bytes);
    }
    historical_policy_detail::unknown(name);
}

} // namespace vllm_kvmem
