// Construction and configuration only; never called by a decode/policy step.
#pragma once

#include "historical_kv_policy.hpp"
#include "historical_redundancy.hpp"

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <cmath>

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
[[noreturn]] inline void unknown(const std::string &name) {
    throw std::invalid_argument("Unknown historical_gc.policy: " + name + " (supported: lru, redundancy)");
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
        else if(key=="importance_weight") config.importance_weight=number(0,1);
        else if(key=="representative_similarity") config.representative_similarity=number(-1,1);
        else if(key=="reference_blocks") config.reference_blocks=integer(65536);
        else if(key=="candidate_blocks") config.candidate_blocks=integer(65536);
        else if(key=="pair_comparisons") config.pair_comparisons=integer(65536);
        else if(key=="query_segments_per_step") config.query_segments_per_step=integer(65536);
        else if(key=="feature_cache_bytes") config.feature_cache_bytes=integer(uint64_t(1)<<40);
        else if(key=="query_cache_bytes") config.query_cache_bytes=integer(uint64_t(1)<<40);
        else throw std::invalid_argument("Unknown historical_gc.policy_params.redundancy parameter: "+key);
    }
    if(!config.keep_mandatory) throw std::invalid_argument("Redundancy GC requires keep_mandatory=true");
    return config;
}
} // namespace historical_policy_detail

inline void validate_historical_kv_policy(const std::string &name,
                                          const HistoricalPolicyParameters &params) {
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

// Add each built-in policy's validation/construction here. Policies only emit
// intents; all implementations share the manager's existing release protocol.
// The fallback preserves the earlier native Config.keep_mandatory API; an
// explicit policy parameter takes precedence.
inline std::unique_ptr<HistoricalKvPolicy> make_historical_kv_policy(
        const std::string &name, const HistoricalPolicyParameters &params,
        size_t capacity, bool keep_mandatory = true) {
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

inline std::shared_ptr<HistoricalFeatureSource> make_historical_feature_source(
        const std::string &name,const HistoricalPolicyParameters &params,size_t capacity) {
    if(name=="lru") {historical_policy_detail::lru_keep_mandatory(params,true);return {};}
    if(name=="redundancy") {
        const auto config=historical_policy_detail::redundancy_config(params);
        return std::make_shared<HistoricalFeatureSource>(capacity,config.feature_cache_bytes,config.query_cache_bytes);
    }
    historical_policy_detail::unknown(name);
}

} // namespace vllm_kvmem
