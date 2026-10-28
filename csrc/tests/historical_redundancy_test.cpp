#include "historical_redundancy.hpp"
#include "historical_policy_factory.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace vllm_kvmem;
using Handle=qw3::KvMemBlockHandle;
static void check(bool value,const char *why) {if(!value) throw std::runtime_error(why);}
static void near(double actual,double expected) {check(std::abs(actual-expected)<1e-5,"numeric feature layout mismatch");}
static std::shared_ptr<HistoricalFeatureSource> source(size_t capacity=8,size_t feature=65536,size_t query=65536) {
    return std::make_shared<HistoricalFeatureSource>(capacity,feature,query);
}
static void question(const std::shared_ptr<HistoricalFeatureSource> &s,const std::vector<float> &q,size_t tokens=1) {
    auto epoch=s->begin_question(tokens);check(s->append_question(epoch,0,q.data(),tokens),"question publication failed");
}
static HistoricalBlockMeta block(const std::shared_ptr<HistoricalFeatureSource> &s,uint32_t slot,const std::vector<float> &k,uint64_t use,bool mandatory=false) {
    HistoricalBlockMeta m; m.handle={slot,1};m.live=m.sealed=true;m.cpu_bytes=1024;m.last_use_seq=m.birth_seq=use;m.logical_id=slot;m.mandatory=mandatory;
    check(s->publish_key(m.handle,k.data(),k.size()),"key publication failed");m.feature_version=s->key_version(m.handle);return m;
}
static std::vector<HistoricalDeleteIntent> decide(HistoricalRedundancyPolicy &policy,const HistoricalSnapshot &snapshot,HistoricalPressure pressure,size_t *steps=nullptr) {
    HistoricalDecisionBatch batch(8);std::vector<HistoricalDeleteIntent> result;
    for(size_t step=0;step<10000;++step) {
        const auto r=policy.step(snapshot,pressure,{1,8},batch);check(r.examined<=1,"metadata work exceeded slice");
        result.insert(result.end(),batch.values().begin(),batch.values().end());
        pressure.pending_slots+=batch.values().size();
        for(const auto &i:batch.values()) { (void)i;pressure.pending_cpu_bytes+=1024; }
        batch.clear_after_ack();
        if(r.status!=HistoricalPolicyStatus::Yielded) {if(steps)*steps=step+1;return result;}
    }
    throw std::runtime_error("redundancy policy did not yield/finish within work bound");
}
static HistoricalPressure pressure() {HistoricalPressure p;p.low_slots=2;p.high_slots=3;return p;}
static HistoricalRedundancyConfig config() {HistoricalRedundancyConfig c;c.reference_blocks=1;c.candidate_blocks=8;c.query_segments_per_step=1;c.pair_comparisons=1;return c;}

static void complete_question_layout_and_observations() {
    auto s=source();s->configure(2,2,4,2);
    auto epoch=s->begin_question(34);check(!s->snapshot().valid(),"incomplete question exposed");
    auto chunk=[](size_t start,size_t tokens) {
        std::vector<float> q(2*tokens*4*2);
        for(size_t l=0;l<2;++l)for(size_t t=0;t<tokens;++t)for(size_t j=0;j<8;++j)q[(l*tokens+t)*8+j]=float(l*1000+(start+t)*10+j);
        return q;
    };
    auto a=chunk(0,13),b=chunk(13,21);
    check(s->append_question(epoch,0,a.data(),13),"first question chunk failed");check(!s->snapshot().valid(),"partial query exposed");
    check(s->append_question(epoch,13,b.data(),21),"second question chunk failed");
    auto snap=s->snapshot();check(snap.valid() && snap.segments()==2,"complete question shape mismatch");
    for(size_t l=0;l<2;++l)for(size_t j=0;j<8;++j) {
        near(snap.segment(0)[l*8+j],l*1000+155+j);near(snap.segment(1)[l*8+j],l*1000+325+j);
    }
    auto obs=chunk(0,32);check(s->observe(epoch,obs.data(),32),"observation append failed");
    auto current=s->snapshot();check(current.segments()==4,"fast8/slow32 windows missing");
    near(current.segment(2)[0],275);near(current.segment(3)[0],155);
    check(snap.segments()==2,"immutable observation snapshot changed");
    auto more=chunk(32,128);check(s->observe(epoch,more.data(),128),"observation ring append failed");
    current=s->snapshot();check(current.segments()==7,"slow ring must keep four windows");
    near(current.segment(2)[0],1555);near(current.segment(3)[0],475);near(current.segment(6)[0],1435);
    s->begin_question(0);check(!s->snapshot().valid(),"zero-token question did not invalidate");
    check(!s->observe(epoch,obs.data(),32),"old observation epoch accepted");
}
static void missing_invalid_immutable_and_budget() {
    auto s=source(4,2048,2048);s->configure(1,1,1,2);
    const float k[]={1,2},changed[]={2,1},nan[]={1,std::numeric_limits<float>::quiet_NaN()},zero[]={0,0};
    check(!s->publish_key({0,1},nan,2),"NaN key accepted");check(!s->publish_key({0,1},zero,2),"zero norm accepted");
    check(s->publish_key({0,1},k,2),"initial key rejected");const auto version=s->key_version({0,1});
    check(s->publish_key({0,1},k,2),"idempotent key rejected");check(!s->publish_key({0,1},changed,2),"sealed key mutation accepted");
    check(s->key_version({0,1})==version,"immutable key version changed");
    auto held=s->key({0,1});const auto bytes=s->stats().at("feature_bytes");s->erase_key({0,1});
    check(s->stats().at("feature_bytes")==bytes,"held key bytes escaped accounting");
    check(!s->key({0,1}),"erased key visible");held.reset();check(s->stats().at("feature_bytes")<bytes,"held key charge not released");
    check(s->publish_key({0,2},changed,2),"reused generation rejected");s->erase_key({0,1});check(s->key_version({0,2})!=0,"stale erase removed new generation");
    std::vector<HistoricalFeatureSource::Snapshot> snapshots;
    bool overflow=false;
    for(size_t i=0;i<64;++i) {
        auto epoch=s->begin_question(1);
        if(!s->append_question(epoch,0,k,1)) {overflow=true;break;}
        snapshots.push_back(s->snapshot());
    }
    check(overflow && s->stats().at("overflow")>0,"retained snapshots not capped");
    check(!s->snapshot().valid(),"overflow did not fail closed");check(s->stats().at("query_bytes")<=2048,"query cap exceeded");
    snapshots.clear();question(s,{1,2});auto epoch=s->snapshot().epoch;
    check(!s->append_question(epoch,0,k,1),"duplicate question accepted");check(!s->snapshot().valid(),"invalid question remained visible");
    question(s,{1,2});epoch=s->snapshot().epoch;check(!s->observe(epoch,nan,1),"NaN observation accepted");check(!s->snapshot().valid(),"NaN observation did not protect");
    s->reset();check(s->stats().at("query_bytes")==0,"reset retained unowned query allocation");
}
static void guards_and_version_revalidation() {
    auto s=source();s->configure(1,1,1,2);auto a=block(s,0,{1,0},1),b=block(s,1,{1,0},2);question(s,{1,0});
    HistoricalDeleteIntent i{a.handle,1,1};i.reason=HistoricalDeleteReason::Redundancy;i.feature_version=a.feature_version;
    i.question_epoch=s->snapshot().epoch;i.witness=b.handle;i.witness_feature_version=b.feature_version;
    auto guard=s->try_commit_guard(i);check(guard.acquired && guard.valid,"valid guarded commit rejected");
    std::atomic<bool> acquired{true};std::thread probe([&]{auto other=s->try_commit_guard(i);acquired=other.acquired;check(s->key_version(a.handle)==0,"contended version lookup blocked/returned mutable data");});probe.join();
    check(!acquired,"contended source guard did not yield");
    std::atomic<bool> began{false},finished{false};
    std::thread invalidate([&]{began=true;s->invalidate_question();finished=true;});
    while(!began)std::this_thread::yield();
    check(!finished,"question mutation crossed guarded retirement interval");
    guard.lock.unlock();invalidate.join();check(finished,"question invalidation did not resume after guard");
    s->begin_question(1);guard=s->try_commit_guard(i);check(guard.acquired && !guard.valid,"incomplete new question did not stale intent");guard.lock.unlock();
    question(s,{1,0});i.question_epoch=s->snapshot().epoch;s->erase_key(b.handle);const float k[]={1,0};s->publish_key({1,2},k,2);
    guard=s->try_commit_guard(i);check(guard.acquired && !guard.valid,"reused witness generation accepted");
}
static void direct_witness_ack_and_nontransitivity() {
    auto run=[](HistoricalCommitCode ack) {
        auto s=source();s->configure(1,1,1,2);
        auto a=block(s,0,{1,0},1),b=block(s,1,{.8660254f,.5f},2);question(s,{1,0});
        auto c=config();c.representative_similarity=.8;HistoricalRedundancyPolicy policy(8,c);policy.bind_features(s);
        HistoricalSnapshot snap{1,1,{a,b}};auto p=pressure();p.high_slots=2;p.low_slots=1;
        const auto intents=decide(policy,snap,p);check(intents.size()==1 && intents[0].handle.slot==0 && intents[0].witness.slot==1,"direct witness not selected");
        policy.on_result(intents[0],ack);
        check(policy.permits(b)==(ack!=HistoricalCommitCode::Committed),"only committed deletion may create carry history");
        auto cc=block(s,2,{.5f,.8660254f},3,true);snap={1,2,{b,cc}};
        const auto next=decide(policy,snap,p);
        if(ack==HistoricalCommitCode::Committed)check(next.empty(),"transitive A~B~C deleted retained direct B witness");
        else check(next.size()==1 && next[0].handle.slot==1,"noncommit incorrectly protected witness forever");
    };
    for(auto ack:{HistoricalCommitCode::Committed,HistoricalCommitCode::Shadow,HistoricalCommitCode::Retry,HistoricalCommitCode::Stale,HistoricalCommitCode::Cancelled})run(ack);
}
static void importance_gqa_head_order_and_segment_peak() {
    auto s=source();s->configure(1,2,4,1);
    auto a=block(s,0,{1,10},2),b=block(s,1,{10,1},1),ref=block(s,7,{1,1},3,true);
    std::vector<float> q(64*4);
    for(size_t t=0;t<64;++t) {const float first[]={2,-5,.1f,-3},second[]={-3,-4,2,-5};std::copy_n(t<32?first:second,4,q.data()+t*4);}
    question(s,q,64);auto c=config();c.importance_weight=1;
    HistoricalRedundancyPolicy policy(8,c);policy.bind_features(s);HistoricalSnapshot snap{1,1,{a,b,ref}};
    auto p=pressure();p.high_slots=2;p.low_slots=1;p.ready_slots=1;p.requested_ready_slots=2;
    size_t steps=0;const auto intents=decide(policy,snap,p,&steps);
    check(steps>=6,"query segments were not sliced across steps");
    check(intents.size()==1 && intents[0].handle.slot==0,"GQA heads or per-segment importance aggregation changed");
}
static void missing_features_fallback_pressure_and_generation() {
    auto s=source();s->configure(1,1,1,2);auto a=block(s,0,{1,0},1),b=block(s,1,{-1,0},2,true);
    auto c=config();c.allow_lossy_fallback=false;HistoricalSnapshot snap{1,1,{a,b}};
    HistoricalRedundancyPolicy failclosed(8,c);failclosed.bind_features(s);
    check(decide(failclosed,snap,pressure()).empty(),"missing question fell back to LRU");
    question(s,{1,0});snap.round=2;check(decide(failclosed,snap,pressure()).empty(),"negative cosine deleted without explicit fallback");
    c.allow_lossy_fallback=true;HistoricalRedundancyPolicy lossy(8,c);lossy.bind_features(s);
    auto p=pressure();p.requested_ready_slots=1;check(decide(lossy,snap,p).empty(),"request readiness allowed lossy fallback");
    snap.round=3;p.requested_ready_slots=0;const auto intents=decide(lossy,snap,p);
    check(intents.size()==1 && intents[0].reason==HistoricalDeleteReason::LossyPressure,"explicit pressure fallback missing reason");
    s->erase_key(a.handle);snap.round=4;check(decide(lossy,snap,p).empty(),"missing key entered lossy fallback");
}
static void factory_validation() {
    check(!make_historical_feature_source("lru",{},8),"LRU allocated redundancy source");
    check(bool(make_historical_feature_source("redundancy",{},8)),"redundancy factory lacks source");
    for(const HistoricalPolicyParameters &params:std::vector<HistoricalPolicyParameters>{
        {{"keep_mandatory",false}},{{"reference_blocks",true}},{{"pair_comparisons",int64_t(0)}},
        {{"query_segments_per_step",int64_t(65537)}},{{"importance_weight",true}},
        {{"importance_weight",std::numeric_limits<double>::infinity()}},{{"unknown",int64_t(1)}},
        {{"representative_similarity",1.1}},{{"feature_cache_bytes",double(1024)}}}) {
        bool failed=false;try{validate_historical_kv_policy("redundancy",params);}catch(const std::invalid_argument &){failed=true;}
        check(failed,"factory accepted invalid redundancy parameter");
    }
    validate_historical_kv_policy("redundancy",{{"representative_similarity",-1.0},{"importance_weight",int64_t(0)}});
}
static void continuous_generations_and_bounded_snapshots() {
    auto s=source(8,4096,4096);s->configure(1,1,2,2);
    const float q[]={1,2,3,4},key[]={1,2};
    std::vector<std::shared_ptr<const HistoricalFeatureSource::Key>> held;
    for(uint64_t generation=1;generation<=128;++generation) {
        const Handle h{uint32_t(generation%8),generation};
        check(s->publish_key(h,key,2),"bounded slot generation reuse failed");
        held.push_back(s->key(h));
        if(held.size()>3)held.erase(held.begin());
        auto epoch=s->begin_question(1);check(s->append_question(epoch,0,q,1),"bounded question reuse failed");
        check(s->observe(epoch,q,1),"bounded accepted Q reuse failed");
        const auto stats=s->stats();check(stats.at("feature_bytes")<=4096 && stats.at("query_bytes")<=4096,"continuous source exceeded cap");
    }
    held.clear();s->reset();const auto bytes=s->stats().at("feature_bytes");
    check(s->stats().at("keys")==0 && s->stats().at("query_bytes")==0,"scope reset left identities/payloads");
    s->configure(1,1,1,1);const float fresh[]={3};check(s->publish_key({0,1},fresh,1),"scope reuse geometry/generation failed");
    s->erase_key({0,1});check(s->stats().at("feature_bytes")==bytes,"feature scope reuse leaked payload");
}
static void pending_credit_pressure_hysteresis_and_no_batch_chains() {
    auto s=source();s->configure(1,1,1,2);question(s,{1,0});
    HistoricalSnapshot snapshot{1,1,{}};
    for(uint32_t slot=0;slot<8;++slot)snapshot.blocks.push_back(block(s,slot,{1,0},slot+1));
    HistoricalRedundancyPolicy policy(8,config());policy.bind_features(s);
    auto p=pressure();p.pending_slots=3;check(!policy.wants_snapshot(p),"pending slots not credited");
    check(decide(policy,snapshot,p).empty(),"pending reserve caused duplicate deletion");
    p.pending_slots=0;check(policy.wants_snapshot(p),"low watermark failed to trigger");
    p.ready_slots=2;check(policy.wants_snapshot(p),"hysteresis stopped below high watermark");
    p.ready_slots=3;check(!policy.wants_snapshot(p),"high watermark failed to stop");
    p.ready_slots=0;snapshot.round=2;const auto intents=decide(policy,snapshot,p);check(intents.size()==3,"bounded policy failed reserve target");
    for(const auto &i:intents)for(const auto &j:intents)
        check(i.witness.slot!=j.handle.slot,"same decision batch deleted its retained witness");
    for(const auto &i:intents)policy.on_result(i,HistoricalCommitCode::Committed);
    check(!policy.permits(snapshot.blocks[intents.front().witness.slot]),"committed witness not carried");
    policy.reset();check(policy.permits(snapshot.blocks[intents.front().witness.slot]),"scope reset retained prior carry");
    check(policy.wants_snapshot(p),"first-round pressure failed to latch");
    p.ready_slots=2;snapshot.round=3;
    check(decide(policy,snapshot,p).size()==1,"snapshot assembly lost first-round hysteresis latch");
}
int main() {
    complete_question_layout_and_observations();missing_invalid_immutable_and_budget();guards_and_version_revalidation();
    direct_witness_ack_and_nontransitivity();importance_gqa_head_order_and_segment_peak();
    missing_features_fallback_pressure_and_generation();factory_validation();
    continuous_generations_and_bounded_snapshots();pending_credit_pressure_hysteresis_and_no_batch_chains();
    std::cout<<"historical redundancy tests passed\n";
}
