// Immutable native mass/exposure state for the existing prefill policy.
// Reuse the exact compensated validation sum, multiply and typed reads from
// the qualified dictionary backend. Only state representation changes.
#include "prefill_attention_commit.hpp"
#include "parallel_rows.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <climits>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>

namespace {
namespace py = pybind11;

// ---- Opt-in KVMEM_CHECKPOINT_FAST_STATE helpers (layerwise_checkpoints.py).
// Text produced here must equal json.dumps(...) of the same Python values so
// that checkpoint evidence digests do not depend on the backend.
inline void append_int(std::string &out,long long value) {
    char buffer[24];
    const auto end=std::to_chars(buffer,buffer+sizeof buffer,value);
    out.append(buffer,end.ptr);
}

#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
#define KVMEM_FLOAT_REPR 1
// float.__repr__: shortest round-trip digits; fixed notation while the decimal
// point position is in (-4, 16], exponent notation (>= 2 exponent digits)
// otherwise; json.dumps spells non-finite values NaN/Infinity/-Infinity.
inline void append_float_repr(std::string &out,double value) {
    if (std::isnan(value)) { out+="NaN";return; }
    if (std::isinf(value)) { out+=value<0?"-Infinity":"Infinity";return; }
    char buffer[48];
    const auto end=std::to_chars(buffer,buffer+sizeof buffer,value,std::chars_format::scientific);
    const char *p=buffer;
    if (*p=='-') { out.push_back('-');++p; }
    char digits[24];int n=0;
    for (;p<end.ptr && *p!='e';++p) if (*p!='.') digits[n++]=*p;
    ++p;
    const bool negative=*p=='-';
    if (*p=='-' || *p=='+') ++p;
    int exponent=0;
    for (;p<end.ptr;++p) exponent=exponent*10+(*p-'0');
    if (negative) exponent=-exponent;
    const int decpt=exponent+1;
    if (decpt>-4 && decpt<=16) {
        if (decpt<=0) {
            out+="0.";out.append(static_cast<size_t>(-decpt),'0');out.append(digits,n);
        } else if (decpt>=n) {
            out.append(digits,n);out.append(static_cast<size_t>(decpt-n),'0');out+=".0";
        } else {
            out.append(digits,decpt);out.push_back('.');out.append(digits+decpt,n-decpt);
        }
    } else {
        out.push_back(digits[0]);
        if (n>1) { out.push_back('.');out.append(digits+1,n-1); }
        out.push_back('e');out.push_back(exponent<0?'-':'+');
        const int magnitude=exponent<0?-exponent:exponent;
        if (magnitude<10) out.push_back('0');
        append_int(out,magnitude);
    }
}
#endif

// A builtin int (no bool, no subclass) that fits a long long.
inline bool exact_int(PyObject *object,long long &value) {
    if (!PyLong_CheckExact(object)) return false;
    int overflow=0;
    value=PyLong_AsLongLongAndOverflow(object,&overflow);
    if (overflow) return false;
    if (value==-1 && PyErr_Occurred()) { PyErr_Clear();return false; }
    return true;
}

// A plain tuple or list: borrowed item array and size.
inline bool plain_sequence(PyObject *object,PyObject **&items,Py_ssize_t &size) {
    if (PyTuple_CheckExact(object)) {
        items=&PyTuple_GET_ITEM(object,0);size=PyTuple_GET_SIZE(object);return true;
    }
    if (PyList_CheckExact(object)) {
        items=PySequence_Fast_ITEMS(object);size=PyList_GET_SIZE(object);return true;
    }
    return false;
}

// (slot, generation, begin, count, dense) as a plain tuple of builtin ints.
inline bool map_row(PyObject *row,long long &begin,long long &count,long long &dense) {
    return PyTuple_CheckExact(row) && PyTuple_GET_SIZE(row)==5 &&
        exact_int(PyTuple_GET_ITEM(row,2),begin) && exact_int(PyTuple_GET_ITEM(row,3),count) &&
        exact_int(PyTuple_GET_ITEM(row,4),dense);
}

// Age-adjusted pressure (strategy_pressure_scorer=lift): distance buckets of the
// expected-attention profile, in tokens from a sampled row back to a block's end:
// [0,32), [32,64), [64,128), ..., [65536,131072), >=131072.
constexpr size_t kLiftBuckets=14;

inline size_t lift_bucket(int64_t distance) {
    if (distance<32) return 0;
    const int log2=63-__builtin_clzll(static_cast<unsigned long long>(distance));
    return std::min<size_t>(kLiftBuckets-1,static_cast<size_t>(log2-4));
}

struct State {
    std::vector<int64_t> keys;
    std::vector<double> mass, exposure;
    std::unordered_map<int64_t,size_t> positions;
    // Lift states also keep, per key, the decayed attention a typical block at the
    // same distance would have received over the same rows, and this layer's
    // running attention profile by distance (this request's own rows). Off: empty.
    bool lift=false;
    std::vector<double> expected;
    std::array<double,kLiftBuckets> bucket_mass{}, bucket_pairs{};

    static State lift_empty() { State result; result.lift=true; return result; }

    // Ranking value for key i: mass / expected; `top` (above every scored key)
    // for keys no row has scored yet, so the newest text ranks first.
    double lift_top() const {
        double best=0.;
        for (size_t i=0;i<keys.size();++i)
            if (expected[i]>0.) best=std::max(best,mass[i]/expected[i]);
        return best+1.;
    }

    static State from_values(const py::dict &values) {
        State result;
        result.keys.reserve(values.size());result.mass.reserve(values.size());
        result.exposure.reserve(values.size());result.positions.reserve(values.size());
        for (const auto &item : values) {
            const auto key=py::cast<int64_t>(item.first);
            const auto pair=py::cast<std::vector<double>>(item.second);
            if (pair.size()!=2) throw std::invalid_argument("Invalid prefill history state");
            result.positions.emplace(key,result.keys.size());result.keys.push_back(key);
            result.mass.push_back(pair[0]);result.exposure.push_back(pair[1]);
        }
        return result;
    }

    // The state from_values(values) builds; a lift state also keeps its distance
    // profile and every remaining key's expected attention. Historical GC prunes a
    // history through this (history.values = {kept keys}), so a lift state may only
    // lose keys: each key must already be here with the same mass and exposure.
    State with_values(const py::dict &values) const {
        State result=from_values(values);
        if (!lift) return result;
        result.lift=true;result.bucket_mass=bucket_mass;result.bucket_pairs=bucket_pairs;
        result.expected.reserve(result.keys.size());
        for (size_t i=0;i<result.keys.size();++i) {
            const auto found=positions.find(result.keys[i]);
            if (found==positions.end() || !(mass[found->second]==result.mass[i]) ||
                    !(exposure[found->second]==result.exposure[i]))
                throw std::invalid_argument("A lift prefill state can only drop keys (a session restore of "
                                            "lift states needs KVMEM_CHECKPOINT_FAST_STATE=1)");
            result.expected.push_back(expected[found->second]);
        }
        return result;
    }

    py::dict values() const {
        py::dict result;
        for (size_t i=0;i<keys.size();++i) {
            py::list pair(2);pair[0]=mass[i];pair[1]=exposure[i];
            result[py::int_(keys[i])]=std::move(pair);
        }
        return result;
    }

    py::dict scores() const {
        // Selection only needs the ratio. Reuse the accepted native doubles
        // without materializing a Python [mass, exposure] list for every key.
        // Keep insertion order, filtering and IEEE division identical to
        // PrefillHistory.scores(); do not quantize or reassociate accumulation.
        py::dict result;
        if (lift && has_lift_score()) {
            const double top=lift_top();
            for (size_t i=0;i<keys.size();++i)
                result[py::int_(keys[i])]=expected[i]>0.?mass[i]/expected[i]:top;
            return result;
        }
        for (size_t i=0;i<keys.size();++i) {
            if (exposure[i]>0.) result[py::int_(keys[i])]=mass[i]/exposure[i];
        }
        return result;
    }

    size_t size() const { return keys.size(); }

    // ---- Opt-in KVMEM_CHECKPOINT_FAST_STATE (layerwise_checkpoints.py).
    bool is_sorted() const {
        for (size_t i=1;i<keys.size();++i) if (keys[i-1]>=keys[i]) return false;
        return true;
    }

    // The state State.from_values({k: v for k, v in sorted(values().items())})
    // builds: the same doubles in ascending key order (keys are unique).
    State sorted() const {
        std::vector<size_t> order(keys.size());
        std::iota(order.begin(),order.end(),size_t(0));
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b) { return keys[a]<keys[b]; });
        State result;
        result.keys.reserve(keys.size());result.mass.reserve(keys.size());
        result.exposure.reserve(keys.size());result.positions.reserve(keys.size());
        result.lift=lift;result.bucket_mass=bucket_mass;result.bucket_pairs=bucket_pairs;
        for (const auto index : order) {
            result.positions.emplace(keys[index],result.keys.size());
            result.keys.push_back(keys[index]);
            result.mass.push_back(mass[index]);result.exposure.push_back(exposure[index]);
            if (lift) result.expected.push_back(expected[index]);
        }
        return result;
    }

    // not any(key >= position for key in values())
    bool keys_below(int64_t position) const {
        for (const auto key : keys) if (key>=position) return false;
        return true;
    }

    // tuple((key, mass, exposure) for key, (mass, exposure) in values().items())
    py::tuple rows() const {
        py::tuple result(keys.size());
        for (size_t i=0;i<keys.size();++i)
            result[i]=py::make_tuple(py::int_(keys[i]),py::float_(mass[i]),py::float_(exposure[i]));
        return result;
    }

#ifdef KVMEM_FLOAT_REPR
    // json.dumps(rows()).encode(), built without the GIL.
    py::bytes rows_json() const {
        std::string out;
        {
            py::gil_scoped_release release;
            out.reserve(keys.size()*56+2);
            out.push_back('[');
            for (size_t i=0;i<keys.size();++i) {
                if (i) out+=", ";
                out.push_back('[');append_int(out,keys[i]);
                out+=", ";append_float_repr(out,mass[i]);
                out+=", ";append_float_repr(out,exposure[i]);
                out.push_back(']');
            }
            out.push_back(']');
        }
        return py::bytes(out);
    }
#endif

    // values() == other.values() as dictionaries (order-insensitive, Python
    // float == semantics: -0.0 == 0.0, NaN != NaN).
    bool same_state(const State &other) const {
        if (other.keys.size()!=keys.size()) return false;
        for (size_t i=0;i<keys.size();++i) {
            const auto found=other.positions.find(keys[i]);
            if (found==other.positions.end()) return false;
            if (!(mass[i]==other.mass[found->second]) || !(exposure[i]==other.exposure[found->second]))
                return false;
        }
        return true;
    }

    // scores().get(key, 0.) for each key, without building the dictionary:
    // the same IEEE division for exposed keys, 0.0 for absent/unexposed ones.
    // Lift scores need at least one positive value; otherwise both lookups keep
    // the original attention / time-visible score.
    bool has_lift_score() const {
        for (size_t i=0;i<keys.size();++i)
            if (expected[i]<=0. || mass[i]>0.) return true;
        return false;
    }

    std::vector<double> scores_for(const std::vector<int64_t> &wanted) const {
        std::vector<double> result(wanted.size(), 0.);
        if (lift && has_lift_score()) {
            const double top=lift_top();
            for (size_t i=0;i<wanted.size();++i) {
                const auto found=positions.find(wanted[i]);
                result[i]=(found!=positions.end() && expected[found->second]>0.)
                    ?mass[found->second]/expected[found->second]:top;
            }
            return result;
        }
        for (size_t i=0;i<wanted.size();++i) {
            const auto found=positions.find(wanted[i]);
            if (found!=positions.end() && exposure[found->second]>0.)
                result[i]=mass[found->second]/exposure[found->second];
        }
        return result;
    }

    // {k: tuple(v) for k, v in values().items()} == {k: tuple(v) for k, v in other.items()}
    // with Python float == semantics (so -0.0 == 0.0 and NaN != NaN).
    bool same_values(const py::dict &other) const {
        if (static_cast<size_t>(py::len(other))!=keys.size()) return false;
        for (const auto &item : other) {
            if (!PyLong_Check(item.first.ptr())) return false;
            int overflow=0;
            const long long key=PyLong_AsLongLongAndOverflow(item.first.ptr(),&overflow);
            if (overflow || (key==-1 && PyErr_Occurred())) { PyErr_Clear(); return false; }
            const auto found=positions.find(static_cast<int64_t>(key));
            if (found==positions.end()) return false;
            const py::handle pair=item.second;
            if (py::len(pair)!=2) return false;
            const double m=py::cast<double>(pair[py::int_(0)]), e=py::cast<double>(pair[py::int_(1)]);
            if (!(m==mass[found->second]) || !(e==exposure[found->second])) return false;
        }
        return true;
    }

    static bool validate_buffer(const py::buffer_info &buffer,
            const std::vector<int64_t> &rows,const std::vector<int64_t> &history) {
        const bool fp32=buffer.format==py::format_descriptor<float>::format() && buffer.itemsize==4;
        const bool fp64=buffer.format==py::format_descriptor<double>::format() && buffer.itemsize==8;
        if (buffer.ndim!=2 || (!fp32 && !fp64))
            throw std::invalid_argument("Prefill attention requires a float32/float64 matrix");
        if (buffer.shape[0]!=static_cast<ssize_t>(rows.size()))
            throw std::runtime_error("Missing prefill attention rows");
        if (buffer.shape[1]!=static_cast<ssize_t>(history.size()))
            throw std::runtime_error("Invalid prefill attention mass");
        if (rows.empty()) throw std::invalid_argument("Prefill attention requires sampled rows");
        return fp32;
    }

    // Called without the GIL by both entry points. Keep each layer's ordered
    // scalar arithmetic identical; batching only shares argument conversion.
    State committed_buffer(double decay,const std::vector<int64_t> &rows,
            const std::vector<int64_t> &starts,const std::vector<int64_t> &history,
            int64_t local_begin,const py::buffer_info &buffer,bool fp32,int64_t history_shift=0) const {
        State result;
        {
            result.keys.reserve(history.size());result.mass.reserve(history.size());
            result.exposure.reserve(history.size());result.positions.reserve(history.size());
            result.lift=lift;
            if (lift) {
                result.expected.reserve(history.size());
                result.bucket_mass=bucket_mass;result.bucket_pairs=bucket_pairs;
            }
            std::vector<size_t> indices;indices.reserve(history.size());
            for (const auto key : history) {
                const auto inserted=result.positions.emplace(key,result.keys.size());
                indices.push_back(inserted.first->second);
                if (!inserted.second)continue;
                result.keys.push_back(key);
                const auto previous=positions.find(key);
                const bool found=previous!=positions.end();
                result.mass.push_back(kvmem_prefill::multiply(decay,found?mass[previous->second]:0.));
                result.exposure.push_back(kvmem_prefill::multiply(decay,found?exposure[previous->second]:0.));
                if (lift)
                    result.expected.push_back(kvmem_prefill::multiply(decay,found?expected[previous->second]:0.));
            }
            const auto read_value=fp32?&kvmem_prefill::read<float>:&kvmem_prefill::read<double>;
            const double inv=1./rows.size();
            std::vector<ssize_t> seen(result.keys.size(),-1);
            if (!lift) {
                // The original loop, unchanged (lift states take the path below).
                for (ssize_t row=0;row<buffer.shape[0];++row) {
                    kvmem_prefill::Sum sum;
                    for (ssize_t page=0;page<buffer.shape[1];++page) {
                        const double value=read_value(buffer,row,page);
                        if (!std::isfinite(value) || value<0.)
                            throw std::runtime_error("Invalid prefill attention mass");
                        sum.add(value);
                    }
                    if (std::abs(sum.value()-1.)>.005)
                        throw std::runtime_error("Invalid prefill attention mass");
                    if ((rows[row]>0 && local_begin>INT64_MAX-rows[row]) ||
                        (rows[row]<0 && local_begin<INT64_MIN-rows[row]))
                        throw std::invalid_argument("Prefill position overflow");
                    const auto position=local_begin+rows[row];
                    for (size_t page=0;page<std::min(starts.size(),history.size());++page) {
                        if (starts[page]>position)continue;
                        const auto index=indices[page];
                        result.mass[index]+=kvmem_prefill::multiply(inv,read_value(buffer,row,page));
                        if (seen[index]!=row) {
                            result.exposure[index]+=inv;seen[index]=row;
                        }
                    }
                }
            } else {
                // Typed reads (no per-value indirect call) for the lift path.
                auto lift_pass=[&](auto tag) {
                    using T=decltype(tag);
                    const char *base=static_cast<const char *>(buffer.ptr);
                    const ssize_t s0=buffer.strides[0],s1=buffer.strides[1];
                    auto rv=[&](ssize_t r,ssize_t c)->double {
                        T value;std::memcpy(&value,base+r*s0+c*s1,sizeof(value));
                        return static_cast<double>(value);
                    };
                    // Lift: pass 1 validates every row (same checks and order) and adds the
                    // packet to this layer's distance profile, counting each row's visible
                    // blocks per distance bucket; pass 2 accumulates mass and exposure as
                    // above plus each block's expected attention: the profile's mean for
                    // its bucket, normalised per row (one table per row, no per-block division).
                    const size_t pages=std::min(starts.size(),history.size());
                    const ssize_t nrows=buffer.shape[0];
                    std::vector<std::array<uint32_t,kLiftBuckets>> counts(static_cast<size_t>(nrows));
                    for (ssize_t row=0;row<nrows;++row) {
                        // One read per value, as in the original validation loop: the
                        // position check moves first so visibility is known while reading.
                        if ((rows[row]>0 && local_begin>INT64_MAX-rows[row]) ||
                            (rows[row]<0 && local_begin<INT64_MIN-rows[row]))
                            throw std::invalid_argument("Prefill position overflow");
                        const auto position=local_begin+rows[row];
                        const auto at=position+history_shift-32;
                        auto &count=counts[static_cast<size_t>(row)];
                        count.fill(0);
                        std::array<double,kLiftBuckets> row_mass{};
                        kvmem_prefill::Sum sum;
                        for (ssize_t page=0;page<buffer.shape[1];++page) {
                            const double value=rv(row,page);
                            if (!std::isfinite(value) || value<0.)
                                throw std::runtime_error("Invalid prefill attention mass");
                            sum.add(value);
                            if (static_cast<size_t>(page)<pages && starts[page]<=position) {
                                const auto bucket=lift_bucket(at-history[page]);
                                row_mass[bucket]+=value;++count[bucket];
                            }
                        }
                        if (std::abs(sum.value()-1.)>.005)
                            throw std::runtime_error("Invalid prefill attention mass");
                        for (size_t b=0;b<kLiftBuckets;++b) {
                            result.bucket_mass[b]+=row_mass[b];result.bucket_pairs[b]+=count[b];
                        }
                    }
                    std::array<double,kLiftBuckets> mean{};
                    for (size_t b=0;b<kLiftBuckets;++b)
                        mean[b]=result.bucket_pairs[b]>0.?result.bucket_mass[b]/std::max(result.bucket_pairs[b],1.):0.;
                    std::array<double,kLiftBuckets> scaled{};
                    for (ssize_t row=0;row<nrows;++row) {
                        const auto position=local_begin+rows[row];
                        const auto at=position+history_shift-32;
                        const auto &count=counts[static_cast<size_t>(row)];
                        double norm=0.;uint32_t visible=0;
                        for (size_t b=0;b<kLiftBuckets;++b) { norm+=count[b]*mean[b];visible+=count[b]; }
                        for (size_t b=0;b<kLiftBuckets;++b)
                            scaled[b]=kvmem_prefill::multiply(inv,norm>0.?mean[b]/norm:1./static_cast<double>(std::max<uint32_t>(visible,1)));
                        for (size_t page=0;page<pages;++page) {
                            if (starts[page]>position)continue;
                            const auto index=indices[page];
                            result.mass[index]+=kvmem_prefill::multiply(inv,rv(row,page));
                            if (seen[index]!=row) {
                                result.exposure[index]+=inv;seen[index]=row;
                            }
                            result.expected[index]+=scaled[lift_bucket(at-history[page])];
                        }
                    }
                };
                if (fp32) lift_pass(float{}); else lift_pass(double{});
            }
        }
        return result;
    }

    State committed(double decay,const std::vector<int64_t> &rows,
            const std::vector<int64_t> &starts,const std::vector<int64_t> &history,
            int64_t local_begin,py::buffer masses,int64_t history_shift) const {
        const auto buffer=masses.request();
        const bool fp32=validate_buffer(buffer,rows,history);
        py::gil_scoped_release release;
        return committed_buffer(decay,rows,starts,history,local_begin,buffer,fp32,history_shift);
    }
};

// The validation committed_batch performs while folding, without folding:
// same buffer/shape checks, finite/nonnegative values, compensated row sums and
// position overflow, in layer then row order, with the same exception types
// and messages. Deferred layer folds (deferred_fold.py) run this at
// publication so an invalid span is still rejected there, before any commit.
void validate_batch(const std::vector<int64_t> &rows,
        const std::vector<std::vector<int64_t>> &histories,
        int64_t local_begin,py::buffer masses) {
    const auto buffer=masses.request();
    const bool fp32=buffer.format==py::format_descriptor<float>::format() && buffer.itemsize==4;
    const bool fp64=buffer.format==py::format_descriptor<double>::format() && buffer.itemsize==8;
    if (buffer.ndim!=3 || (!fp32 && !fp64))
        throw std::invalid_argument("Prefill attention requires a float32/float64 matrix");
    if (buffer.shape[0]!=static_cast<ssize_t>(histories.size()))
        throw std::invalid_argument("Missing independent prefill attention layers");
    std::vector<py::buffer_info> layers;
    layers.reserve(histories.size());
    for (size_t layer=0;layer<histories.size();++layer)
        layers.emplace_back(static_cast<char *>(buffer.ptr)+static_cast<ssize_t>(layer)*buffer.strides[0],
            buffer.itemsize,buffer.format,2,
            std::vector<ssize_t>{buffer.shape[1],buffer.shape[2]},
            std::vector<ssize_t>{buffer.strides[1],buffer.strides[2]},buffer.readonly);
    py::gil_scoped_release release;
    // Same layer parallelism as committed_batch (KVMEM_NATIVE_PUBLISH_THREADS);
    // for_rows rethrows the first failing layer in layer order.
    kvmem_parallel::for_rows(histories.size(),[&](size_t layer) {
        const auto &view=layers[layer];
        const bool layer_fp32=State::validate_buffer(view,rows,histories[layer]);
        const auto read_value=layer_fp32?&kvmem_prefill::read<float>:&kvmem_prefill::read<double>;
        for (ssize_t row=0;row<view.shape[0];++row) {
            kvmem_prefill::Sum sum;
            for (ssize_t page=0;page<view.shape[1];++page) {
                const double value=read_value(view,row,page);
                if (!std::isfinite(value) || value<0.)
                    throw std::runtime_error("Invalid prefill attention mass");
                sum.add(value);
            }
            if (std::abs(sum.value()-1.)>.005)
                throw std::runtime_error("Invalid prefill attention mass");
            if ((rows[row]>0 && local_begin>INT64_MAX-rows[row]) ||
                (rows[row]<0 && local_begin<INT64_MIN-rows[row]))
                throw std::invalid_argument("Prefill position overflow");
        }
    });
}

std::vector<State> committed_batch(py::iterable state_objects,
        const std::vector<double> &decays,const std::vector<int64_t> &rows,
        const std::vector<int64_t> &starts,
        const std::vector<std::vector<int64_t>> &histories,
        int64_t local_begin,py::buffer masses,int64_t history_shift) {
    // Own a private immutable container before extracting pointers. The caller
    // may supply a generator or mutate its original list while the GIL is free.
    const py::tuple owners(state_objects);
    std::vector<const State *> states;
    states.reserve(owners.size());
    for (const auto owner : owners) states.push_back(py::cast<const State *>(owner));
    const auto buffer=masses.request();
    const bool fp32=buffer.format==py::format_descriptor<float>::format() && buffer.itemsize==4;
    const bool fp64=buffer.format==py::format_descriptor<double>::format() && buffer.itemsize==8;
    if (buffer.ndim!=3 || (!fp32 && !fp64))
        throw std::invalid_argument("Prefill attention requires a float32/float64 matrix");
    if (decays.size()!=states.size() || histories.size()!=states.size() ||
            buffer.shape[0]!=static_cast<ssize_t>(states.size()))
        throw std::invalid_argument("Missing independent prefill attention layers");
    for (const auto state : states)
        if (!state) throw std::invalid_argument("Missing native prefill attention state");

    // These descriptors borrow the original buffer, including signed strides.
    // The Python arguments and buffer lease stay alive until the GIL is held
    // again; no per-layer Tensor/numpy views or score copies are necessary.
    std::vector<py::buffer_info> layers;
    layers.reserve(states.size());
    for (size_t layer=0;layer<states.size();++layer)
        layers.emplace_back(static_cast<char *>(buffer.ptr)+static_cast<ssize_t>(layer)*buffer.strides[0],
            buffer.itemsize,buffer.format,2,
            std::vector<ssize_t>{buffer.shape[1],buffer.shape[2]},
            std::vector<ssize_t>{buffer.strides[1],buffer.strides[2]},buffer.readonly);
    std::vector<State> result(states.size());
    {
        py::gil_scoped_release release;
        // Layers are independent: each keeps its own ordered scalar arithmetic.
        // Opt-in KVMEM_NATIVE_PUBLISH_THREADS spreads them across threads; the
        // first failing layer (in layer order) is rethrown and nothing returns.
        kvmem_parallel::for_rows(states.size(),[&](size_t layer) {
            const bool layer_fp32=State::validate_buffer(layers[layer],rows,histories[layer]);
            result[layer]=states[layer]->committed_buffer(decays[layer],rows,starts,
                histories[layer],local_begin,layers[layer],layer_fp32,history_shift);
        });
    }
    return result;
}

// ---- Opt-in KVMEM_CHECKPOINT_FAST_STATE (layerwise_checkpoints.py).
// Every function returns None for inputs outside the plain shapes it handles
// (tuples of builtin ints); the caller then runs the original Python code.

// The per-row loop and the length/partial-geometry checks of
// layerwise_checkpoints._resolve: same checks, order and messages.
// maps: tuple of tuples of (slot, generation, begin, count, dense);
// by_begin: {history_begin: record}. Returns (chosen or None, canonical).
py::object resolve_rows(py::object maps,py::dict by_begin,long long position,
        long long block_tokens,bool want_chosen) {
    if (!PyTuple_CheckExact(maps.ptr())) return py::none();
    struct Entry {
        PyObject *record=nullptr;          // borrowed from by_begin
        py::object slot,generation;
        long long slot_value=0,generation_value=0,count=0;
        bool ints=false;
        Py_ssize_t stamp=0;                // 1 + last layer that used it
    };
    std::unordered_map<long long,Entry> memo;
    const Py_ssize_t layers=PyTuple_GET_SIZE(maps.ptr());
    py::tuple canonical(layers),chosen(want_chosen?layers:0);
    std::vector<long long> totals;
    std::vector<std::vector<std::array<long long,3>>> partial(layers);
    for (Py_ssize_t layer=0;layer<layers;++layer) {
        PyObject *rows=PyTuple_GET_ITEM(maps.ptr(),layer);
        if (!PyTuple_CheckExact(rows)) return py::none();
        const Py_ssize_t count_rows=PyTuple_GET_SIZE(rows);
        py::tuple mapped(count_rows),picked(want_chosen?count_rows:0);
        long long cursor=0;
        for (Py_ssize_t i=0;i<count_rows;++i) {
            PyObject *row=PyTuple_GET_ITEM(rows,i);
            long long begin,count,dense;
            if (!map_row(row,begin,count,dense)) return py::none();
            auto found=memo.find(begin);
            const bool seen=found!=memo.end() && found->second.stamp==layer+1;
            if (seen || dense!=cursor || begin<0 || count<=0 ||
                    count>LLONG_MAX-begin || begin+count>position)
                throw py::value_error("Invalid layerwise checkpoint coordinates");
            if (found==memo.end()) {
                PyObject *record=PyDict_GetItemWithError(by_begin.ptr(),PyTuple_GET_ITEM(row,2));
                if (!record) {
                    if (PyErr_Occurred()) throw py::error_already_set();
                    throw py::value_error("Layerwise checkpoint source was not retained");
                }
                if (record==Py_None)
                    throw py::value_error("Layerwise checkpoint source was not retained");
                Entry entry;
                entry.record=record;
                const py::object record_count=py::handle(record).attr("count");
                if (!exact_int(record_count.ptr(),entry.count)) return py::none();
                if (entry.count!=count)
                    throw py::value_error("Layerwise checkpoint source was not retained");
                const py::object handle=py::handle(record).attr("handle");
                entry.slot=handle.attr("slot");entry.generation=handle.attr("generation");
                entry.ints=exact_int(entry.slot.ptr(),entry.slot_value) &&
                    exact_int(entry.generation.ptr(),entry.generation_value);
                found=memo.emplace(begin,std::move(entry)).first;
            }
            Entry &entry=found->second;
            if (entry.count!=count)
                throw py::value_error("Layerwise checkpoint source was not retained");
            entry.stamp=layer+1;
            long long slot,generation;
            if (entry.ints && exact_int(PyTuple_GET_ITEM(row,0),slot) &&
                    exact_int(PyTuple_GET_ITEM(row,1),generation) &&
                    slot==entry.slot_value && generation==entry.generation_value) {
                Py_INCREF(row);PyTuple_SET_ITEM(mapped.ptr(),i,row);    // already canonical
            } else {
                PyObject *fresh=PyTuple_Pack(5,entry.slot.ptr(),entry.generation.ptr(),
                    PyTuple_GET_ITEM(row,2),PyTuple_GET_ITEM(row,3),PyTuple_GET_ITEM(row,4));
                if (!fresh) throw py::error_already_set();
                PyTuple_SET_ITEM(mapped.ptr(),i,fresh);
            }
            if (want_chosen) {
                Py_INCREF(entry.record);PyTuple_SET_ITEM(picked.ptr(),i,entry.record);
            }
            if (count!=block_tokens) partial[layer].push_back({begin,count,dense});
            if (cursor>LLONG_MAX-count) return py::none();
            cursor+=count;
        }
        totals.push_back(cursor);
        PyTuple_SET_ITEM(canonical.ptr(),layer,mapped.release().ptr());
        if (want_chosen) PyTuple_SET_ITEM(chosen.ptr(),layer,picked.release().ptr());
    }
    if (totals.empty() || std::any_of(totals.begin(),totals.end(),
            [&](long long total) { return total!=totals[0]; }))
        throw py::value_error("Layerwise checkpoint has inconsistent physical lengths");
    for (const auto &rows : partial)
        if (rows!=partial[0])
            throw py::value_error("Layerwise checkpoint has inconsistent partial-block geometry");
    py::object picked_all=py::none();
    if (want_chosen) picked_all=std::move(chosen);
    return py::make_tuple(std::move(picked_all),std::move(canonical));
}

// _identity(a) == _identity(b) for plain rows: (begin, count, dense) per row.
py::object same_identity(py::object a,py::object b) {
    PyObject **left,**right;Py_ssize_t left_size,right_size;
    if (!plain_sequence(a.ptr(),left,left_size) || !plain_sequence(b.ptr(),right,right_size))
        return py::none();
    std::vector<std::array<long long,3>> first(left_size),second(right_size);
    for (Py_ssize_t i=0;i<left_size;++i)
        if (!map_row(left[i],first[i][0],first[i][1],first[i][2])) return py::none();
    for (Py_ssize_t i=0;i<right_size;++i)
        if (!map_row(right[i],second[i][0],second[i][1],second[i][2])) return py::none();
    return py::bool_(first==second);
}

// sorted({(row[0], row[1]) for group in groups for row in group}), optionally
// without the (-1, -1) placeholder (layerwise_gc_holds.window_handles).
py::object unique_handles(py::object groups,bool drop_placeholder) {
    PyObject **outer;Py_ssize_t outer_size;
    if (!plain_sequence(groups.ptr(),outer,outer_size)) return py::none();
    std::vector<std::pair<long long,long long>> pairs;
    for (Py_ssize_t g=0;g<outer_size;++g) {
        PyObject **rows;Py_ssize_t count;
        if (!plain_sequence(outer[g],rows,count)) return py::none();
        pairs.reserve(pairs.size()+count);
        for (Py_ssize_t i=0;i<count;++i) {
            PyObject *row=rows[i];
            long long slot,generation;
            if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row)<2 ||
                    !exact_int(PyTuple_GET_ITEM(row,0),slot) ||
                    !exact_int(PyTuple_GET_ITEM(row,1),generation))
                return py::none();
            if (drop_placeholder && slot==-1 && generation==-1) continue;
            pairs.emplace_back(slot,generation);
        }
    }
    std::sort(pairs.begin(),pairs.end());
    pairs.erase(std::unique(pairs.begin(),pairs.end()),pairs.end());
    py::list result(pairs.size());
    for (size_t i=0;i<pairs.size();++i)
        result[i]=py::make_tuple(py::int_(pairs[i].first),py::int_(pairs[i].second));
    return result;
}

// (json.dumps(_identity(rows)).encode(), sum(row[3] for row in rows)).
py::object identity_json(py::object rows) {
    PyObject **items;Py_ssize_t size;
    if (!plain_sequence(rows.ptr(),items,size)) return py::none();
    std::string out;
    out.reserve(static_cast<size_t>(size)*24+2);
    out.push_back('[');
    long long total=0;
    for (Py_ssize_t i=0;i<size;++i) {
        long long begin,count,dense;
        if (!map_row(items[i],begin,count,dense)) return py::none();
        if ((count>0 && total>LLONG_MAX-count) || (count<0 && total<LLONG_MIN-count))
            return py::none();
        total+=count;
        if (i) out+=", ";
        out.push_back('[');append_int(out,begin);out+=", ";append_int(out,count);
        out+=", ";append_int(out,dense);out.push_back(']');
    }
    out.push_back(']');
    return py::make_tuple(py::bytes(out),py::int_(total));
}
} // namespace

PYBIND11_MODULE(_prefill_state_native,module) {
    py::class_<State>(module,"State")
        .def_static("from_values",&State::from_values)
        .def("with_values",&State::with_values)
        .def("values",&State::values)
        .def("scores",&State::scores)
        .def("scores_for",&State::scores_for)
        .def("same_values",&State::same_values)
        .def("same_state",&State::same_state)
        .def("size",&State::size)
        .def("is_sorted",&State::is_sorted)
        .def("sorted",&State::sorted)
        .def("keys_below",&State::keys_below)
        .def("rows",&State::rows)
#ifdef KVMEM_FLOAT_REPR
        .def("rows_json",&State::rows_json)
#endif
        .def("committed",&State::committed,py::arg("decay"),py::arg("rows"),py::arg("starts"),
             py::arg("history"),py::arg("local_begin"),py::arg("masses"),py::arg("history_shift")=0)
        .def_static("lift_empty",&State::lift_empty)
        .def_readonly("lift",&State::lift);
    module.def("committed_batch",&committed_batch,py::arg("states"),py::arg("decays"),py::arg("rows"),
               py::arg("starts"),py::arg("histories"),py::arg("local_begin"),py::arg("masses"),
               py::arg("history_shift")=0);
    module.def("validate_batch",&validate_batch);
    module.def("resolve_rows",&resolve_rows);
    module.def("same_identity",&same_identity);
    module.def("unique_handles",&unique_handles);
    module.def("identity_json",&identity_json);
}
