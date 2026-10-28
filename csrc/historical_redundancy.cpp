#include "historical_redundancy.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>

namespace vllm_kvmem {
namespace {
using Handle = qw3::KvMemBlockHandle;
bool same(Handle a, Handle b) { return a.slot == b.slot && a.generation == b.generation; }
size_t mul(size_t a, size_t b) {
    if (a && b > SIZE_MAX / a) throw std::invalid_argument("Historical feature size overflow");
    return a * b;
}
size_t add(size_t a, size_t b) {
    if (b > SIZE_MAX - a) throw std::invalid_argument("Historical feature size overflow");
    return a + b;
}
struct Budget {
    explicit Budget(size_t n) : limit(n) {}
    size_t limit;
    std::atomic<size_t> used{0}, peak{0}, overflow{0};
};
struct Charge {
    std::shared_ptr<Budget> budget;
    size_t bytes;
    ~Charge() { budget->used.fetch_sub(bytes); }
};
std::shared_ptr<void> charge(const std::shared_ptr<Budget> &b, size_t bytes) {
    size_t old = b->used.load();
    do {
        if (bytes > b->limit || old > b->limit - bytes) { ++b->overflow; return {}; }
    } while (!b->used.compare_exchange_weak(old, old + bytes));
    size_t peak = b->peak.load();
    while (peak < old + bytes && !b->peak.compare_exchange_weak(peak, old + bytes)) {}
    try { return std::shared_ptr<void>(new Charge{b, bytes}); }
    catch (...) { b->used.fetch_sub(bytes); throw; }
}
bool finite(const float *p, size_t n) {
    return p && std::all_of(p, p + n, [](float x) { return std::isfinite(x); });
}
uint64_t deficit(uint64_t target, uint64_t free, uint64_t pending) {
    const auto missing = target > free ? target - free : 0;
    return missing > pending ? missing - pending : 0;
}
auto age(const HistoricalBlockMeta &b) {
    return std::make_tuple(b.last_use_seq, b.birth_seq, b.logical_id,
                           b.handle.slot, b.handle.generation);
}
} // namespace

struct HistoricalFeatureSource::State {
    mutable std::mutex mu;
    std::shared_ptr<Budget> features, queries;
    std::shared_ptr<void> index_charge, rings_charge;
    std::vector<std::shared_ptr<const Key>> keys;
    std::array<size_t, 4> shape{};
    uint64_t version = 0, epoch = 0, invalid = 0, key_count = 0;
    mutable std::atomic<uint64_t> missing{0};
    size_t expected = 0, accepted = 0;
    bool ready = false;
    std::shared_ptr<QueryBank> building;
    std::shared_ptr<const QueryBank> question, observations;
    std::vector<float> fast_ring, slow_ring;
    std::vector<double> slow_sum;
    size_t observed = 0, slow_count = 0, slow_head = 0;
    size_t key_width() const { return shape[0] * shape[1] * shape[3]; }
    size_t query_width() const { return shape[0] * shape[2] * shape[3]; }
    void invalidate() {
        ++epoch;
        ready = false; expected = accepted = observed = slow_count = slow_head = 0;
        building.reset(); question.reset(); observations.reset();
        std::fill(fast_ring.begin(), fast_ring.end(), 0.f);
        std::fill(slow_ring.begin(), slow_ring.end(), 0.f);
        std::fill(slow_sum.begin(), slow_sum.end(), 0.);
    }
    std::shared_ptr<QueryBank> bank(size_t segments, bool sums) {
        const size_t width = query_width();
        const size_t bytes = add(sizeof(QueryBank) + 128,
            add(mul(mul(segments, width), sizeof(float)), sums ? mul(width, sizeof(double)) : 0));
        auto ticket = charge(queries, bytes);
        if (!ticket) return {};
        auto result = std::make_shared<QueryBank>();
        result->charge = std::move(ticket);
        result->width = width; result->segments = segments;
        result->values.resize(mul(segments, width));
        if (sums) result->sums.resize(width);
        return result;
    }
};

HistoricalFeatureSource::HistoricalFeatureSource(size_t capacity, size_t feature_bytes, size_t query_bytes)
    : state_(std::make_unique<State>()) {
    if (!capacity || !feature_bytes || !query_bytes)
        throw std::invalid_argument("Historical feature capacities must be positive");
    auto &s = *state_;
    s.features = std::make_shared<Budget>(feature_bytes);
    s.queries = std::make_shared<Budget>(query_bytes);
    s.index_charge = charge(s.features, add(256, mul(capacity, sizeof(std::shared_ptr<const Key>))));
    if (!s.index_charge) throw std::invalid_argument("Historical feature budget cannot hold slot index");
    s.keys.resize(capacity);
}
HistoricalFeatureSource::~HistoricalFeatureSource() = default;
void HistoricalFeatureSource::configure(size_t l, size_t k, size_t q, size_t d) {
    if (!l || l > 4 || !k || k > 4096 || !q || q > 4096 || q < k || q % k || !d || d > 4096)
        throw std::invalid_argument("Invalid historical feature geometry");
    auto &s = *state_;
    std::lock_guard<std::mutex> lock(s.mu);
    const std::array<size_t, 4> shape{l,k,q,d};
    if (s.shape[0]) {
        if (s.shape != shape) throw std::invalid_argument("Historical feature geometry is immutable per scope");
        return;
    }
    const size_t width = mul(l, mul(q, d));
    auto ticket = charge(s.queries, add(128, mul(width, 12 * sizeof(float) + sizeof(double))));
    if (!ticket) throw std::invalid_argument("Historical query budget cannot hold observation rings");
    std::vector<float> fast(mul(8,width)), slow(mul(4,width));
    std::vector<double> sums(width);
    s.fast_ring = std::move(fast); s.slow_ring = std::move(slow); s.slow_sum = std::move(sums);
    s.rings_charge = std::move(ticket); s.shape = shape;
}
std::array<size_t,4> HistoricalFeatureSource::geometry() const {
    std::lock_guard<std::mutex> lock(state_->mu); return state_->shape;
}
size_t HistoricalFeatureSource::key_elements() const {
    std::lock_guard<std::mutex> lock(state_->mu); return state_->key_width();
}
size_t HistoricalFeatureSource::query_elements_per_token() const {
    std::lock_guard<std::mutex> lock(state_->mu); return state_->query_width();
}
bool HistoricalFeatureSource::publish_key(Handle handle, const float *p, size_t count) {
    const auto shape = geometry();
    const size_t groups = shape[0] * shape[1], dim = shape[3];
    if (!handle.generation || handle.slot >= state_->keys.size() || !groups || count != groups*dim || !finite(p,count))
        return false;
    // An unchanged sealed publication needs no second allocation, including
    // when the feature budget is already fully occupied.
    {
        std::lock_guard<std::mutex> lock(state_->mu);
        const auto &old=state_->keys[handle.slot];
        if(old && same(old->handle,handle))
            return old->values.size()==count && std::equal(old->values.begin(),old->values.end(),p);
    }
    // All allocation, copying and norm calculation are outside the source commit gate.
    auto ticket = charge(state_->features, add(sizeof(Key)+128, add(mul(count,sizeof(float)),mul(groups,sizeof(double)))));
    if (!ticket) return false;
    auto row = std::make_shared<Key>();
    row->charge = std::move(ticket); row->handle = handle;
    row->values.assign(p,p+count); row->inverse_norms.resize(groups);
    for (size_t g=0; g<groups; ++g) {
        double n=0;
        for (size_t d=0;d<dim;++d) n += double(p[g*dim+d])*p[g*dim+d];
        if (!(n>0) || !std::isfinite(n)) return false;
        row->inverse_norms[g] = 1/std::sqrt(n);
    }
    auto &s = *state_;
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.shape != shape) return false;
    auto &old = s.keys[handle.slot];
    if (old && same(old->handle,handle)) return old->values == row->values;
    row->version = ++s.version;
    if (!old) ++s.key_count;
    old = std::move(row);
    return true;
}
uint64_t HistoricalFeatureSource::begin_question(size_t expected) {
    auto &s=*state_;
    std::lock_guard<std::mutex> lock(s.mu);
    s.invalidate();
    if (!s.shape[0] || !expected || expected > SIZE_MAX-31) { ++s.invalid; return s.epoch; }
    try { s.building=s.bank((expected+31)/32,true); }
    catch (const std::bad_alloc &) { ++s.queries->overflow; }
    catch (const std::invalid_argument &) { ++s.queries->overflow; }
    if (!s.building) { ++s.invalid; return s.epoch; }
    s.expected=expected;
    return s.epoch;
}
bool HistoricalFeatureSource::append_question(uint64_t epoch,size_t offset,const float *p,size_t tokens) {
    auto &s=*state_;
    std::lock_guard<std::mutex> lock(s.mu);
    if (epoch != s.epoch) return false;
    const size_t width=s.query_width(), l=s.shape[0], h=s.shape[2], d=s.shape[3];
    if (!s.building || s.ready || !tokens || offset!=s.accepted || tokens>s.expected-s.accepted ||
            !width || tokens>SIZE_MAX/width || !finite(p,tokens*width)) { ++s.invalid; s.invalidate(); return false; }
    for (size_t t=0;t<tokens;++t) {
        const size_t pos=s.accepted+t;
        for (size_t layer=0;layer<l;++layer)
            for(size_t j=0;j<h*d;++j)
                s.building->sums[layer*h*d+j] += p[(layer*tokens+t)*h*d+j];
        if ((pos+1)%32==0 || pos+1==s.expected) {
            const double n=double(pos%32+1);
            float *dst=s.building->values.data()+(pos/32)*width;
            for(size_t j=0;j<width;++j) { dst[j]=float(s.building->sums[j]/n); s.building->sums[j]=0; }
        }
    }
    s.accepted+=tokens;
    if(s.accepted==s.expected) { s.question=s.building; s.building.reset(); s.ready=true; }
    return true;
}
bool HistoricalFeatureSource::observe(uint64_t epoch,const float *p,size_t tokens) {
    auto &s=*state_;
    std::lock_guard<std::mutex> lock(s.mu);
    if(epoch!=s.epoch) return false;
    const size_t width=s.query_width(), l=s.shape[0], hd=s.shape[2]*s.shape[3];
    if(!s.ready || !tokens || !width || tokens>SIZE_MAX/width || !finite(p,tokens*width) || tokens>SIZE_MAX-s.observed) {
        ++s.invalid; s.invalidate(); return false;
    }
    // The immutable next observation owns its allocation before old snapshots can
    // be released. Insufficient space invalidates this question, never drops Q.
    std::shared_ptr<QueryBank> next;
    try { next=s.bank(5,false); } catch(const std::bad_alloc &) { ++s.queries->overflow; }
    if(!next) { ++s.invalid; s.invalidate(); return false; }
    for(size_t t=0;t<tokens;++t) {
        for(size_t layer=0;layer<l;++layer) for(size_t j=0;j<hd;++j) {
            const size_t dst=layer*hd+j;
            const float value=p[(layer*tokens+t)*hd+j];
            s.fast_ring[(s.observed%8)*width+dst]=value;
            s.slow_sum[dst]+=value;
        }
        ++s.observed;
        if(s.observed%32==0) {
            for(size_t j=0;j<width;++j) { s.slow_ring[s.slow_head*width+j]=float(s.slow_sum[j]/32); s.slow_sum[j]=0; }
            s.slow_head=(s.slow_head+1)%4; s.slow_count=std::min<size_t>(4,s.slow_count+1);
        }
    }
    size_t segments=0;
    if(s.observed>=8) {
        for(size_t j=0;j<width;++j) { double v=0; for(size_t t=0;t<8;++t) v+=s.fast_ring[t*width+j]; next->values[j]=float(v/8); }
        ++segments;
    }
    for(size_t row=0;row<s.slow_count;++row) {
        const size_t slot=(s.slow_head+4-s.slow_count+row)%4;
        std::copy_n(s.slow_ring.data()+slot*width,width,next->values.data()+segments*width);
        ++segments;
    }
    next->segments=segments;
    s.observations=std::move(next);
    return true;
}
void HistoricalFeatureSource::invalidate_question() {
    std::lock_guard<std::mutex> lock(state_->mu); ++state_->invalid; state_->invalidate();
}
void HistoricalFeatureSource::erase_key(Handle h) {
    auto &s=*state_; std::lock_guard<std::mutex> lock(s.mu);
    if(h.slot<s.keys.size() && s.keys[h.slot] && same(s.keys[h.slot]->handle,h)) {
        s.keys[h.slot].reset(); --s.key_count;
    }
}
void HistoricalFeatureSource::reset() {
    auto &s=*state_; std::lock_guard<std::mutex> lock(s.mu);
    s.invalidate();
    for(auto &key:s.keys) key.reset();
    s.key_count=0; s.shape={};
    std::vector<float>().swap(s.fast_ring); std::vector<float>().swap(s.slow_ring);
    std::vector<double>().swap(s.slow_sum); s.rings_charge.reset();
}
std::shared_ptr<const HistoricalFeatureSource::Key> HistoricalFeatureSource::key(Handle h) const {
    auto &s=*state_; std::lock_guard<std::mutex> lock(s.mu);
    if(h.slot<s.keys.size() && s.keys[h.slot] && same(s.keys[h.slot]->handle,h)) return s.keys[h.slot];
    ++s.missing; return {};
}
uint64_t HistoricalFeatureSource::key_version(Handle h) const {
    auto &s=*state_; std::unique_lock<std::mutex> lock(s.mu,std::try_to_lock);
    if(!lock.owns_lock()) return 0;
    if(h.slot<s.keys.size() && s.keys[h.slot] && same(s.keys[h.slot]->handle,h)) return s.keys[h.slot]->version;
    ++s.missing; return 0;
}
HistoricalFeatureSource::Snapshot HistoricalFeatureSource::snapshot() const {
    auto &s=*state_; std::lock_guard<std::mutex> lock(s.mu);
    if(!s.ready || !s.question) { ++s.missing; return {}; }
    return {s.epoch,s.shape,s.question,s.observations};
}
const float *HistoricalFeatureSource::Snapshot::segment(size_t i) const {
    if(!question || i>=segments()) throw std::out_of_range("Historical query segment");
    if(i<question->segments) return question->values.data()+i*question->width;
    return observations->values.data()+(i-question->segments)*observations->width;
}
HistoricalFeatureSource::CommitGuard HistoricalFeatureSource::try_commit_guard(const HistoricalDeleteIntent &i) const {
    CommitGuard guard; auto &s=*state_;
    guard.lock=std::unique_lock<std::mutex>(s.mu,std::try_to_lock);
    guard.acquired=guard.lock.owns_lock();
    if(!guard.acquired) return guard;
    auto valid_key=[&](Handle h,uint64_t version) {
        return h.generation && version && h.slot<s.keys.size() && s.keys[h.slot] &&
            same(s.keys[h.slot]->handle,h) && s.keys[h.slot]->version==version;
    };
    guard.valid=s.ready && s.question && i.question_epoch==s.epoch && valid_key(i.handle,i.feature_version);
    if(i.reason==HistoricalDeleteReason::Redundancy)
        guard.valid=guard.valid && !same(i.handle,i.witness) && valid_key(i.witness,i.witness_feature_version);
    return guard;
}
std::map<std::string,uint64_t> HistoricalFeatureSource::stats() const {
    auto &s=*state_; std::lock_guard<std::mutex> lock(s.mu);
    return {{"feature_bytes",s.features->used.load()},{"feature_peak_bytes",s.features->peak.load()},
        {"feature_capacity_bytes",s.features->limit},{"query_bytes",s.queries->used.load()},
        {"query_peak_bytes",s.queries->peak.load()},{"query_capacity_bytes",s.queries->limit},
        {"missing",s.missing.load()},{"overflow",s.features->overflow.load()+s.queries->overflow.load()},
        {"invalid_questions",s.invalid},{"keys",s.key_count},{"question_epoch",s.epoch},
        {"question_tokens",s.accepted},{"question_segments",s.ready?s.question->segments:0},
        {"observation_segments",s.observations?s.observations->segments:0},{"observed_tokens",s.observed},
        {"question_ready",s.ready?1u:0u}};
}

struct HistoricalRedundancyPolicy::State {
    struct Seen { uint64_t generation=0, version=0; };
    struct Row {
        HistoricalBlockMeta meta;
        std::shared_ptr<const HistoricalFeatureSource::Key> key;
        double importance=-std::numeric_limits<double>::infinity(), similarity=-1, score=0;
        size_t witness=SIZE_MAX;
        bool reference=false, valid=false;
    };
    size_t capacity;
    HistoricalRedundancyConfig config;
    std::shared_ptr<const HistoricalFeatureSource> source;
    HistoricalFeatureSource::Snapshot query;
    std::vector<Seen> carry;
    std::vector<uint64_t> reference_generations;
    std::vector<HistoricalBlockMeta> candidates, refs;
    std::vector<Row> rows;
    std::vector<size_t> ranked, fallback;
    uint64_t incarnation=0,round=0;
    size_t cursor=0,origin=0,score_row=0,score_segment=0,pair_row=0,pair_ref=0,emit_cursor=0,fallback_cursor=0;
    bool initialized=false,done=false,collected=false,scored=false,paired=false;
    bool slots_pressure=false,bytes_pressure=false;
    HistoricalPolicyStatus terminal=HistoricalPolicyStatus::NoCandidate;
    State(size_t n,HistoricalRedundancyConfig c):capacity(n),config(c),carry(n),reference_generations(n) {
        static_assert(sizeof(Seen)+sizeof(uint64_t)<=64,"Update admission metadata charge");
        static_assert(sizeof(HistoricalBlockMeta)+sizeof(Row)+2*sizeof(size_t)<=512,"Update admission pool charge");
        candidates.reserve(std::min(n,c.candidate_blocks)); refs.reserve(std::min(n,c.reference_blocks));
        rows.reserve(std::min(n,c.candidate_blocks)+std::min(n,c.reference_blocks));
        ranked.reserve(std::min(n,c.candidate_blocks)); fallback.reserve(std::min(n,c.candidate_blocks));
    }
    bool carries(const HistoricalBlockMeta &m) const {
        return m.handle.slot<carry.size() && carry[m.handle.slot].generation==m.handle.generation &&
            carry[m.handle.slot].version && carry[m.handle.slot].version==m.feature_version;
    }
    bool eligible(const HistoricalBlockMeta &m) const {
        return m.live && m.sealed && m.handle.generation && !m.mandatory && !carries(m);
    }
    bool reference_better(const HistoricalBlockMeta &a,const HistoricalBlockMeta &b) const {
        return std::make_tuple(a.mandatory,carries(a),age(a)) > std::make_tuple(b.mandatory,carries(b),age(b));
    }
    void pressure(const HistoricalPressure &p) {
        if(!p.high_slots || p.ready_slots>=p.high_slots) slots_pressure=false;
        else if(p.ready_slots<p.low_slots) slots_pressure=true;
        if(!p.high_cpu_bytes || p.free_cpu_bytes>=p.high_cpu_bytes) bytes_pressure=false;
        else if(p.free_cpu_bytes<p.low_cpu_bytes) bytes_pressure=true;
    }
    std::pair<uint64_t,uint64_t> needed(const HistoricalPressure &p) const {
        return {std::max(slots_pressure?deficit(p.high_slots,p.ready_slots,p.pending_slots):0,
                         deficit(p.requested_ready_slots,p.ready_slots,p.pending_slots)),
            std::max(bytes_pressure?deficit(p.high_cpu_bytes,p.free_cpu_bytes,p.pending_cpu_bytes):0,
                     deficit(p.scope_cpu_bytes,p.scope_cpu_limit,p.scope_pending_cpu_bytes))};
    }
    void clear_round() {
        for(const auto &m:refs) reference_generations[m.handle.slot]=0;
        query={}; candidates.clear(); refs.clear(); rows.clear(); ranked.clear(); fallback.clear();
        cursor=score_row=score_segment=pair_row=pair_ref=emit_cursor=fallback_cursor=0;
        collected=scored=paired=done=false;
    }
    HistoricalPolicyResult finish(HistoricalPolicyStatus status,size_t examined=0,size_t emitted=0) {
        done=true; terminal=status;
        // Old immutable source banks must not stay charged until another pressure episode.
        query={}; rows.clear();
        return {status,examined,emitted};
    }
    double importance(const HistoricalFeatureSource::Key &key,const float *q) const {
        const auto g=query.geometry;
        const size_t l=g[0],kh=g[1],qh=g[2],d=g[3],ratio=qh/kh;
        double value=0;
        for(size_t layer=0;layer<l;++layer) for(size_t head=0;head<kh;++head) {
            double highest=-std::numeric_limits<double>::infinity();
            const float *k=key.values.data()+(layer*kh+head)*d;
            for(size_t group=0;group<ratio;++group) {
                double dot=0;
                const float *query_head=q+(layer*qh+head*ratio+group)*d;
                for(size_t j=0;j<d;++j) dot+=double(k[j])*query_head[j];
                highest=std::max(highest,dot);
            }
            value+=highest;
        }
        return value/(double(l*kh)*std::sqrt(double(d)));
    }
    double cosine(const HistoricalFeatureSource::Key &a,const HistoricalFeatureSource::Key &b) const {
        const size_t groups=query.geometry[0]*query.geometry[1],d=query.geometry[3];
        double total=0;
        for(size_t g=0;g<groups;++g) {
            double dot=0;
            for(size_t j=0;j<d;++j) dot+=double(a.values[g*d+j])*b.values[g*d+j];
            total+=dot*a.inverse_norms[g]*b.inverse_norms[g];
        }
        return std::clamp(total/double(groups),-1.,1.);
    }
    bool witness_better(size_t a,size_t b) const {
        if(b==SIZE_MAX) return true;
        return std::make_tuple(rows[a].meta.mandatory,carries(rows[a].meta),rows[a].importance,age(rows[a].meta)) >
               std::make_tuple(rows[b].meta.mandatory,carries(rows[b].meta),rows[b].importance,age(rows[b].meta));
    }
};

HistoricalRedundancyPolicy::HistoricalRedundancyPolicy(size_t capacity,HistoricalRedundancyConfig config)
    : state_(std::make_unique<State>(capacity,config)) {
    const auto valid_count=[](size_t n){return n && n<=65536;};
    if(!capacity || !config.keep_mandatory || !std::isfinite(config.importance_weight) ||
            config.importance_weight<0 || config.importance_weight>1 ||
            !std::isfinite(config.representative_similarity) || config.representative_similarity< -1 ||
            config.representative_similarity>1 || !valid_count(config.reference_blocks) ||
            !valid_count(config.candidate_blocks) || !valid_count(config.pair_comparisons) ||
            !valid_count(config.query_segments_per_step) || !config.feature_cache_bytes ||
            config.feature_cache_bytes>(uint64_t(1)<<40) || !config.query_cache_bytes ||
            config.query_cache_bytes>(uint64_t(1)<<40))
        throw std::invalid_argument("Invalid historical redundancy policy configuration");
}
HistoricalRedundancyPolicy::~HistoricalRedundancyPolicy()=default;
void HistoricalRedundancyPolicy::bind_features(std::shared_ptr<const HistoricalFeatureSource> source) {
    if(!source) throw std::invalid_argument("Redundancy GC needs native features");
    state_->source=std::move(source);
}
void HistoricalRedundancyPolicy::reset() {
    auto &s=*state_; s.clear_round();
    s.initialized=false; s.slots_pressure=s.bytes_pressure=false; s.origin=0;
    std::fill(s.carry.begin(),s.carry.end(),State::Seen{});
}
bool HistoricalRedundancyPolicy::wants_snapshot(const HistoricalPressure &p) {
    auto &s=*state_; s.pressure(p); auto [slots,bytes]=s.needed(p); return slots || bytes;
}
bool HistoricalRedundancyPolicy::permits(const HistoricalBlockMeta &b) const {return state_->eligible(b);}
HistoricalPolicyResult HistoricalRedundancyPolicy::step(const HistoricalSnapshot &snapshot,
        const HistoricalPressure &p,HistoricalWorkBudget budget,HistoricalDecisionBatch &out) {
    auto &s=*state_;
    if(snapshot.blocks.size()>s.capacity || !budget.scan_blocks || !budget.delete_blocks ||
            p.requested_ready_slots>s.capacity || ((p.low_slots || p.high_slots) &&
            (!p.low_slots || p.low_slots>=p.high_slots || p.high_slots>s.capacity)) ||
            ((p.low_cpu_bytes || p.high_cpu_bytes) && (!p.low_cpu_bytes || p.low_cpu_bytes>=p.high_cpu_bytes)))
        throw std::invalid_argument("Invalid historical redundancy work/pressure bounds");
    if(!out.empty()) return {HistoricalPolicyStatus::Backpressure,0,0};
    if(!s.initialized || s.incarnation!=snapshot.incarnation || s.round!=snapshot.round) {
        if(s.initialized && !s.done) throw std::logic_error("Historical snapshot changed before redundancy round completion");
        if(!s.initialized || s.incarnation!=snapshot.incarnation) {
            std::fill(s.carry.begin(),s.carry.end(),State::Seen{});
            // wants_snapshot may already have latched low-water pressure.
            // Preserve it if ready capacity moved into the hysteresis band
            // while this first metadata snapshot was being assembled.
            if(s.initialized) s.slots_pressure=s.bytes_pressure=false;
        }
        s.clear_round(); s.initialized=true; s.incarnation=snapshot.incarnation; s.round=snapshot.round;
        if(s.source) s.query=s.source->snapshot();
    }
    if(s.done) return {s.terminal,0,0};
    s.pressure(p); auto [slots,bytes]=s.needed(p);
    if(!slots && !bytes) return s.finish(HistoricalPolicyStatus::Complete);
    if(!s.source || !s.query.valid() || s.source->snapshot().epoch!=s.query.epoch)
        return s.finish(HistoricalPolicyStatus::NoCandidate);
    size_t examined=0;
    if(!s.collected) {
        // One bounded rotating candidate pool plus a newest/mandatory reference
        // heap. References are retained for this whole round, preventing chains
        // or A->B/B->C within a single decision batch.
        const size_t limit=std::min(s.config.reference_blocks,std::max<size_t>(1,s.capacity/4));
        const auto compare=[&](const auto &a,const auto &b){return s.reference_better(a,b);};
        while(s.cursor<snapshot.blocks.size() && examined<budget.scan_blocks) {
            const auto &m=snapshot.blocks[(s.origin+s.cursor++)%snapshot.blocks.size()]; ++examined;
            if(!m.live || !m.sealed || !m.handle.generation || m.handle.slot>=s.capacity || !m.feature_version) continue;
            auto &seen=s.carry[m.handle.slot];
            if(seen.generation!=m.handle.generation || seen.version!=m.feature_version) seen={};
            if(s.refs.size()<limit) {s.refs.push_back(m);std::push_heap(s.refs.begin(),s.refs.end(),compare);}
            else if(s.reference_better(m,s.refs.front())) {
                std::pop_heap(s.refs.begin(),s.refs.end(),compare);s.refs.back()=m;std::push_heap(s.refs.begin(),s.refs.end(),compare);
            }
            if(s.eligible(m) && s.candidates.size()<s.config.candidate_blocks) s.candidates.push_back(m);
        }
        if(s.cursor<snapshot.blocks.size()) return {HistoricalPolicyStatus::Yielded,examined,0};
        auto append=[&](const HistoricalBlockMeta &m,bool reference) {
            auto key=s.source->key(m.handle);
            const bool valid=key && key->version==m.feature_version;
            s.rows.push_back({m,std::move(key),-std::numeric_limits<double>::infinity(),-1,0,SIZE_MAX,reference,valid});
        };
        for(const auto &m:s.refs) {s.reference_generations[m.handle.slot]=m.handle.generation;append(m,true);}
        for(const auto &m:s.candidates) {
            if(s.reference_generations[m.handle.slot]!=m.handle.generation) append(m,false);
        }
        s.collected=true;
        s.origin=snapshot.blocks.empty()?0:(s.origin+s.config.candidate_blocks)%snapshot.blocks.size();
    }
    size_t scored=0;
    while(s.score_row<s.rows.size() && scored<s.config.query_segments_per_step) {
        auto &row=s.rows[s.score_row];
        if(!row.valid) {++s.score_row;s.score_segment=0;continue;}
        const double value=s.importance(*row.key,s.query.segment(s.score_segment++)); ++scored;
        if(!std::isfinite(value)) {row.valid=false;++s.score_row;s.score_segment=0;continue;}
        row.importance=std::max(row.importance,value);
        if(s.score_segment==s.query.segments()) {++s.score_row;s.score_segment=0;}
    }
    if(s.score_row<s.rows.size()) return {HistoricalPolicyStatus::Yielded,examined,0};
    size_t pairs=0;
    while(s.pair_row<s.rows.size() && pairs<s.config.pair_comparisons) {
        auto &row=s.rows[s.pair_row];
        if(row.reference || !row.valid) {++s.pair_row;s.pair_ref=0;continue;}
        while(s.pair_ref<s.refs.size() && pairs<s.config.pair_comparisons) {
            const size_t ref=s.pair_ref++; ++pairs;
            if(!s.rows[ref].valid) continue;
            const double cosine=s.cosine(*row.key,*s.rows[ref].key);
            if(!std::isfinite(cosine)) continue;
            row.similarity=std::max(row.similarity,cosine);
            if(cosine>=s.config.representative_similarity && s.witness_better(ref,row.witness)) row.witness=ref;
        }
        if(s.pair_ref==s.refs.size()) {++s.pair_row;s.pair_ref=0;}
    }
    if(s.pair_row<s.rows.size()) return {HistoricalPolicyStatus::Yielded,examined,0};
    if(!s.paired) {
        double max_importance=-std::numeric_limits<double>::infinity(),max_similarity=-1;
        for(size_t i=0;i<s.rows.size();++i) {
            auto &row=s.rows[i];
            if(row.reference || !row.valid) continue;
            if(row.witness!=SIZE_MAX) {s.ranked.push_back(i);max_importance=std::max(max_importance,row.importance);max_similarity=std::max(max_similarity,row.similarity);}
            else s.fallback.push_back(i);
        }
        double sum_importance=0,sum_similarity=0;
        for(auto i:s.ranked) {sum_importance+=std::exp(s.rows[i].importance-max_importance);sum_similarity+=std::exp(s.rows[i].similarity-max_similarity);}
        for(auto i:s.ranked) s.rows[i].score=s.config.importance_weight*std::exp(s.rows[i].importance-max_importance)/sum_importance-
            (1-s.config.importance_weight)*std::exp(s.rows[i].similarity-max_similarity)/sum_similarity;
        std::sort(s.ranked.begin(),s.ranked.end(),[&](size_t a,size_t b){return std::make_tuple(s.rows[a].score,age(s.rows[a].meta))<std::make_tuple(s.rows[b].score,age(s.rows[b].meta));});
        std::sort(s.fallback.begin(),s.fallback.end(),[&](size_t a,size_t b){return std::make_tuple(s.rows[a].importance,age(s.rows[a].meta))<std::make_tuple(s.rows[b].importance,age(s.rows[b].meta));});
        s.paired=true;
    }
    size_t emitted=0;
    while((slots||bytes) && emitted<budget.delete_blocks && out.remaining()) {
        size_t index=SIZE_MAX;
        HistoricalDeleteReason reason=HistoricalDeleteReason::Redundancy;
        if(s.emit_cursor<s.ranked.size()) index=s.ranked[s.emit_cursor++];
        else if(s.config.allow_lossy_fallback && !p.requested_ready_slots && s.fallback_cursor<s.fallback.size()) {
            index=s.fallback[s.fallback_cursor++];reason=HistoricalDeleteReason::LossyPressure;
        }
        if(index==SIZE_MAX) return s.finish(HistoricalPolicyStatus::NoCandidate,examined,emitted);
        const auto &row=s.rows[index];
        if(!s.eligible(row.meta)) continue;
        HistoricalDeleteIntent intent{row.meta.handle,snapshot.incarnation,snapshot.round};
        intent.reason=reason;intent.feature_version=row.key->version;intent.question_epoch=s.query.epoch;
        if(reason==HistoricalDeleteReason::Redundancy) {
            const auto &w=s.rows[row.witness];intent.witness=w.meta.handle;intent.witness_feature_version=w.key->version;
        }
        out.push(intent);++emitted;if(slots)--slots;bytes=bytes>row.meta.cpu_bytes?bytes-row.meta.cpu_bytes:0;
    }
    if(!slots&&!bytes) return s.finish(HistoricalPolicyStatus::Complete,examined,emitted);
    return {HistoricalPolicyStatus::Yielded,examined,emitted};
}
void HistoricalRedundancyPolicy::on_result(const HistoricalDeleteIntent &i,HistoricalCommitCode code) noexcept {
    if(code!=HistoricalCommitCode::Committed || i.reason!=HistoricalDeleteReason::Redundancy) return;
    auto &s=*state_;
    if(i.incarnation!=s.incarnation || i.witness.slot>=s.carry.size() || !i.witness.generation || !i.witness_feature_version) return;
    s.carry[i.witness.slot]={i.witness.generation,i.witness_feature_version};
}

} // namespace vllm_kvmem
