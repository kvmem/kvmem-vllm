// Host-only bindings. Original algorithms remain in third_party/qw3.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "qw3/kvmem_store.hpp"
#include "qw3/nvme_kv_tier.hpp"
#include "qw3/kvmem_repository_gc.hpp"
#include "historical_payload.hpp"
#include "historical_manager.hpp"
#include "thinking_bindings.hpp"
#include "private_record_pack.hpp"
#include "kvmem_request_plan.hpp"
#include "kvmem_prefix_reuse_policy.hpp"
#include "kvmem_query_history.hpp"
#include "kvmem_action_attention.hpp"
#include "prefill_attention_commit.hpp"
#include "block_graph_accept.hpp"
#include "decode_attention_commit.hpp"
#include "graph_propagation.hpp"
#include "action_download_buffer.hpp"
#include "restore_frame.hpp"
#include "action_window_segments.hpp"
#include "query_key_fps_cpu.hpp"
#include "parallel_rows.hpp"
#include "session_spill.hpp"
#include <filesystem>
#include <thread>
#include <mutex>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <limits>
#include <array>
#include <cmath>
#include <chrono>
#include <sys/stat.h>

namespace py = pybind11;
using namespace qw3;

// Tensor views retain this Python buffer exporter. The storage itself is a
// plain C++ allocation; its deleter never owns or calls back into Python/Torch.
struct NativeIndexStorage {
    std::unique_ptr<uint8_t[]> data;
};
struct NativeIndexOwner {
    uint8_t *data = nullptr;
    size_t bytes = 0;
    std::shared_ptr<vllm_kvmem::HistoricalPayloadArena::ExternalOwner> owner;
};

static py::buffer_info historical_feature_buffer(py::buffer buffer, const std::vector<size_t> &shape) {
    auto info = buffer.request();
    if (info.itemsize != sizeof(float) || info.format != py::format_descriptor<float>::format() ||
            size_t(info.ndim) != shape.size())
        throw std::invalid_argument("Historical features require a CPU float32 buffer with the configured shape");
    size_t stride = sizeof(float);
    for (size_t axis = shape.size(); axis-- > 0;) {
        if (!shape[axis] || info.shape[axis] < 0 || size_t(info.shape[axis]) != shape[axis] ||
                (shape[axis] > 1 && (info.strides[axis] < 0 || size_t(info.strides[axis]) != stride)) ||
                stride > SIZE_MAX / shape[axis])
            throw std::invalid_argument("Historical features require a C-contiguous buffer with the configured shape");
        stride *= shape[axis];
    }
    return info;
}

// A private, named inode allows strict reservation verification without changing
// the imported tier. Unlink at construction exit (including exceptions), leaving
// the tier's own fd as the sole owner, exactly as for its ephemeral arena.
struct ReservationFile {
    int fd = -1;
    std::string path;
    explicit ReservationFile(const std::string &directory) {
        std::filesystem::create_directories(directory);
        std::string pattern = directory + "/kvmem-reserve-XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        fd = ::mkstemp(name.data());
        if (fd < 0) throw std::runtime_error("Cannot create private NVMe reservation file");
        path = name.data();
    }
    ~ReservationFile() {
        if (fd >= 0) ::close(fd);
        if (!path.empty()) ::unlink(path.c_str());
    }
    ReservationFile(const ReservationFile &) = delete;
    ReservationFile &operator=(const ReservationFile &) = delete;
};

// vector<move-only Lease> can falsely satisfy is_copy_constructible on libstdc++.
// Give pybind11 an explicitly non-copyable owner for asynchronous read leases.
struct OwnedRead : KvMemStorageRead {
    explicit OwnedRead(KvMemStorageRead &&r) : KvMemStorageRead(std::move(r)) {}
    OwnedRead(const OwnedRead &) = delete;
    OwnedRead(OwnedRead &&) = default;
};

struct OwnedScoreFrame {
    KvMemRepositoryScoreFrame frame;
    std::vector<bool> score_mask, transport_mask;
    OwnedScoreFrame() = default;
    OwnedScoreFrame(const OwnedScoreFrame &) = delete;
    void close() { frame.leases.clear(); }
};

struct ScopedSelectionBudget {
    KvMemStore &store;
    uint32_t previous;
    ScopedSelectionBudget(KvMemStore &s, uint32_t budget)
        : store(s), previous(s.runtime_select_budget_tokens()) {
        if (budget) store.set_runtime_select_budget(budget);
    }
    ~ScopedSelectionBudget() { store.set_runtime_select_budget(previous); }
};

// Python callbacks run only on the CPU policy worker. Joining it while holding
// the GIL would deadlock a pending release callback. Store's keep_alive binding
// below retains this bridge as long as native Store retains its GC instance.
struct RepositoryGcBridge {
    py::function release;
    std::shared_ptr<KvMemRepositoryGc> native;
    std::mutex stop_mutex, error_mutex;
    std::string callback_error;

    RepositoryGcBridge(KvMemRepositoryGc::Config config,
                       std::shared_ptr<KvMemRepository> repository, py::function callback)
        : release(std::move(callback)) {
        native = std::make_shared<KvMemRepositoryGc>(config, std::move(repository),
            [this](const KvMemRepository::Record &record) {
                py::gil_scoped_acquire gil;
                try {
                    return release(py::cast(record, py::return_value_policy::copy)).cast<bool>();
                } catch (py::error_already_set &error) {
                    const std::string message = error.what();
                    {
                        std::lock_guard<std::mutex> guard(error_mutex);
                        if (callback_error.empty()) callback_error = message;
                    }
                    // The native worker records failures and retains Releasing
                    // generations for retry. Surface the error on the foreground.
                    throw std::runtime_error(message);
                }
            });
    }
    void stop() {
        py::gil_scoped_release no_gil;
        std::lock_guard<std::mutex> guard(stop_mutex);
        native->stop();
    }
    ~RepositoryGcBridge() {
        stop();
        native.reset();  // Callback and py::function destruct with the GIL held.
    }
    std::string last_error() {
        std::lock_guard<std::mutex> guard(error_mutex);
        return callback_error;
    }
};

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KVMEM_AVX2 __attribute__((target("avx2")))
static bool native_decode_sum_supported() { return __builtin_cpu_supports("avx2"); }
#else
#define KVMEM_AVX2
static bool native_decode_sum_supported() { return false; }
#endif

static inline float bf16_as_float(const char *source) {
    uint16_t bf16;
    std::memcpy(&bf16, source, 2);
    const uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}

static inline const std::array<float, 256> &fp8_value_table() {
    // CUDA __nv_fp8_e4m3 / torch.float8_e4m3fn, unit scale. Include signed
    // zero, subnormals and both NaN codes; finite source bytes widen exactly.
    static const std::array<float, 256> values = [] {
        std::array<float, 256> table{};
        for (unsigned code=0; code<256; ++code) {
            const int exponent=(code>>3)&15, mantissa=code&7;
            float value=(exponent==15 && mantissa==7)
                ? std::numeric_limits<float>::quiet_NaN()
                : exponent==0 ? std::ldexp(static_cast<float>(mantissa), -9)
                              : std::ldexp(static_cast<float>(8+mantissa), exponent-10);
            table[code]=(code&128) ? -value : value;
        }
        return table;
    }();
    return values;
}

static inline float fp8_as_float(const char *source) {
    return fp8_value_table()[static_cast<unsigned char>(*source)];
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
KVMEM_AVX2 static inline __m256 decode_fp8_e4m3_8(const char *row) {
    const __m256i codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
        reinterpret_cast<const __m128i *>(row)));
    const __m256i magnitude = _mm256_and_si256(codes, _mm256_set1_epi32(127));
    // Normal E4M3: exponent bias 7 -> FP32 bias 127; the three mantissa
    // bits become FP32's top three fraction bits. All conversions are exact.
    const __m256 normal = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(magnitude, _mm256_set1_epi32(120*8)), 20));
    const __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(magnitude),
                                         _mm256_set1_ps(1.f/512.f));
    const __m256 small = _mm256_castsi256_ps(_mm256_cmpgt_epi32(
        _mm256_set1_epi32(8), magnitude));
    __m256 value = _mm256_blendv_ps(normal, subnormal, small);
    const __m256 nan = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
        magnitude, _mm256_set1_epi32(127)));
    value = _mm256_blendv_ps(value, _mm256_set1_ps(fp8_value_table()[127]), nan);
    const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(codes,
        _mm256_set1_epi32(128)), 24);
    return _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(value), sign));
}

KVMEM_AVX2 static void finalize_mean_fp8_simd(const py::buffer_info &raw,
                                            const py::buffer_info &sums, ssize_t count) {
    // Independent channels use eight lanes. Each lane still adds tokens in
    // their original order from +0, with no horizontal reduction or FMA.
    const float *table = fp8_value_table().data();
    const ssize_t channels = sums.shape[1] * sums.shape[2];
    const ssize_t token_stride = raw.strides[1];
    for (ssize_t layer = 0; layer < sums.shape[0]; ++layer) {
        const char *base = static_cast<const char *>(raw.ptr) + layer * raw.strides[0];
        float *out = reinterpret_cast<float *>(static_cast<char *>(sums.ptr) + layer * sums.strides[0]);
        ssize_t channel = 0;
        // Four independent accumulators hide conversion latency without changing
        // the dependency chain (or operand order) of any individual channel.
        for (; channel + 32 <= channels; channel += 32) {
            __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
            const char *row = base + channel;
            for (ssize_t token = 0; token < count; ++token, row += token_stride) {
                s0 = _mm256_add_ps(s0, decode_fp8_e4m3_8(row));
                s1 = _mm256_add_ps(s1, decode_fp8_e4m3_8(row+8));
                s2 = _mm256_add_ps(s2, decode_fp8_e4m3_8(row+16));
                s3 = _mm256_add_ps(s3, decode_fp8_e4m3_8(row+24));
            }
            _mm256_storeu_ps(out+channel, s0);
            _mm256_storeu_ps(out+channel+8, s1);
            _mm256_storeu_ps(out+channel+16, s2);
            _mm256_storeu_ps(out+channel+24, s3);
        }
        for (; channel + 8 <= channels; channel += 8) {
            __m256 total = _mm256_setzero_ps();
            const char *row = base + channel;
            for (ssize_t token = 0; token < count; ++token, row += token_stride) {
                total = _mm256_add_ps(total, decode_fp8_e4m3_8(row));
            }
            _mm256_storeu_ps(out + channel, total);
        }
        for (; channel < channels; ++channel) {
            float total = 0.f;
            const char *row = base + channel;
            for (ssize_t token = 0; token < count; ++token, row += token_stride)
                total += table[static_cast<unsigned char>(*row)];
            out[channel] = total;
        }
    }
}
#endif

static bool mean_fp8_simd_eligible(const py::buffer_info &raw,
                                 const py::buffer_info &sums, ssize_t count) {
    if (!native_decode_sum_supported() || sums.shape[1] * sums.shape[2] < 8)
        return false;
    // The scalar API historically accepts overlapping buffers. Changing its
    // write order for those unusual callers would change subsequent reads.
    const uintptr_t source = reinterpret_cast<uintptr_t>(raw.ptr);
    const uintptr_t destination = reinterpret_cast<uintptr_t>(sums.ptr);
    const size_t source_bytes = (sums.shape[0]-1)*raw.strides[0] +
        (count-1)*raw.strides[1] + sums.shape[1]*sums.shape[2];
    const size_t destination_bytes = sums.shape[0]*sums.strides[0];
    return source + source_bytes <= destination || destination + destination_bytes <= source;
}

// True when every written buffer is pairwise disjoint from all other written
// buffers and from every read buffer. Only then may independent rows run on
// several threads without changing any byte a serial loop would produce.
static bool disjoint_buffers(const std::vector<const py::buffer_info *> &writes,
                             const std::vector<const py::buffer_info *> &reads = {}) {
    auto range = [](const py::buffer_info &v, std::pair<uintptr_t, uintptr_t> &out) {
        if (!v.ptr) return true;
        size_t bytes = static_cast<size_t>(v.itemsize);
        for (ssize_t axis = 0; axis < v.ndim; ++axis) {
            if (v.shape[axis] <= 0) return true;
            if (v.strides[axis] < 0) return false;
            bytes += static_cast<size_t>(v.shape[axis] - 1) * static_cast<size_t>(v.strides[axis]);
        }
        const auto begin = reinterpret_cast<uintptr_t>(v.ptr);
        out = {begin, begin + bytes};
        return true;
    };
    std::vector<std::pair<uintptr_t, uintptr_t>> written;
    written.reserve(writes.size());
    for (const auto *v : writes) {
        std::pair<uintptr_t, uintptr_t> r{0, 0};
        if (!range(*v, r)) return false;
        if (r.second > r.first) written.push_back(r);
    }
    std::sort(written.begin(), written.end());
    for (size_t i = 1; i < written.size(); ++i)
        if (written[i].first < written[i - 1].second) return false;
    for (const auto *v : reads) {
        std::pair<uintptr_t, uintptr_t> r{0, 0};
        if (!range(*v, r)) return false;
        if (r.second <= r.first) continue;
        // First written range whose end is beyond this read's start.
        auto it = std::lower_bound(written.begin(), written.end(), r,
            [](const std::pair<uintptr_t, uintptr_t> &w, const std::pair<uintptr_t, uintptr_t> &x) {
                return w.second <= x.first;
            });
        if (it != written.end() && it->first < r.second) return false;
    }
    return true;
}

template<int Count, bool Fp8=false>
KVMEM_AVX2 static void add_decode_mean(const py::buffer_info &raw,
                                     const py::buffer_info &sums, ssize_t begin) {
    const auto width = sums.shape[1] * sums.shape[2];
    auto *out = static_cast<float *>(sums.ptr);
    for (ssize_t layer = 0; layer < sums.shape[0]; ++layer) {
        const auto *base = static_cast<const char *>(raw.ptr) + layer * raw.strides[0] + begin * raw.strides[1];
        auto *destination = out + layer * width;
        for (ssize_t column = 0; column < width; ++column) {
            float total = 0.f;
            // Compile-time count permits vectorization across columns without
            // reassociating the FP32 addition order along the token dimension.
            for (int token = 0; token < Count; ++token)
                if constexpr (Fp8)
                    total += fp8_as_float(base + token * raw.strides[1] + column);
                else
                    total += bf16_as_float(base + token * raw.strides[1] + column * 2);
            destination[column] += total;
        }
    }
}

// Opt-in (KVMEM_PRESSURE_NATIVE_READER + adopt): one prepared reader's
// repository pins held natively, instead of one Python Lease per record.
// Same pins and lifetime; reset()/clear() releases them all.
struct LeaseSet {
    std::vector<KvMemRepository::Lease> leases;
    LeaseSet() = default;
    LeaseSet(const LeaseSet &) = delete;
    LeaseSet &operator=(const LeaseSet &) = delete;
    LeaseSet(LeaseSet &&) = default;
    LeaseSet &operator=(LeaseSet &&) = default;
    bool valid() const {
        for (const auto &lease : leases) if (!lease) return false;
        return true;
    }
    void reset() {
        for (auto &lease : leases) lease.reset();
        leases.clear();
    }
};

// Opt-in KVMEM_GRAPH_NATIVE_SIZE: block_graph._size in C++. Same traversal
// rules (tuple/list items, dict keys and values, dataclass fields, every
// object counted once by id via the caller's ``seen`` set) and the same
// per-object size function as sys.getsizeof, so the total is identical.
static size_t graph_size_impl(PyObject *value, PyObject *seen, PyObject *fields) {
    PyObject *key = PyLong_FromVoidPtr(value);
    if (!key) throw py::error_already_set();
    const int present = PySet_Contains(seen, key);
    if (present < 0) { Py_DECREF(key); throw py::error_already_set(); }
    if (present) { Py_DECREF(key); return 0; }
    if (PySet_Add(seen, key) < 0) { Py_DECREF(key); throw py::error_already_set(); }
    Py_DECREF(key);
    const size_t own = _PySys_GetSizeOf(value);
    if (own == static_cast<size_t>(-1) && PyErr_Occurred()) throw py::error_already_set();
    size_t result = own;
    if (PyTuple_Check(value) || PyList_Check(value)) {
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(value);
        PyObject **items = PySequence_Fast_ITEMS(value);
        for (Py_ssize_t i = 0; i < n; ++i) result += graph_size_impl(items[i], seen, fields);
    } else if (PyDict_Check(value)) {
        PyObject *k, *v; Py_ssize_t pos = 0;
        while (PyDict_Next(value, &pos, &k, &v)) {
            Py_INCREF(k); Py_INCREF(v);
            try { result += graph_size_impl(k, seen, fields); result += graph_size_impl(v, seen, fields); }
            catch (...) { Py_DECREF(k); Py_DECREF(v); throw; }
            Py_DECREF(k); Py_DECREF(v);
        }
    } else {
        const int dataclass = PyObject_HasAttrString(value, "__dataclass_fields__");
        if (dataclass) {
            py::object listed = py::reinterpret_borrow<py::object>(fields)(py::handle(value));
            for (const auto &field : listed) {
                py::object attribute = py::reinterpret_borrow<py::object>(value).attr(field.attr("name"));
                result += graph_size_impl(attribute.ptr(), seen, fields);
            }
        }
    }
    return result;
}

// Opt-in KVMEM_PRESSURE_NATIVE_READER: one prepared pressure reader's
// immutable record sources, exported once. gather() copies one region of each
// indexed record into a destination slice, record after record (the layout of
// MemoryArchive._gather_region), with the GIL released for the whole copy.
class RecordSources {
public:
    RecordSources(const std::vector<py::buffer> &sources, size_t record_bytes) : record_bytes_(record_bytes) {
        if (!record_bytes) throw std::invalid_argument("Invalid prepared record size");
        sources_.reserve(sources.size());
        for (const auto &source : sources) {
            auto view = source.request();
            if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1 ||
                static_cast<size_t>(view.size) != record_bytes)
                throw std::invalid_argument("Prepared CPU regions require exact immutable record bytes");
            sources_.push_back(std::move(view));
        }
    }
    size_t size() const { return sources_.size(); }
    // Opt-in KVMEM_PRESSURE_READER_EXTEND: append the sources of a lazily
    // selected layer. Every new source is validated before any is appended,
    // so a rejected batch leaves the existing indices and sources unchanged.
    size_t extend(const std::vector<py::buffer> &sources) {
        std::vector<py::buffer_info> views;
        views.reserve(sources.size());
        for (const auto &source : sources) {
            auto view = source.request();
            if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1 ||
                static_cast<size_t>(view.size) != record_bytes_)
                throw std::invalid_argument("Prepared CPU regions require exact immutable record bytes");
            views.push_back(std::move(view));
        }
        const size_t base = sources_.size();
        sources_.reserve(base + views.size());
        for (auto &view : views) sources_.push_back(std::move(view));
        return base;
    }
    void check(const std::vector<int64_t> &indices) const {
        if (indices.empty()) throw py::value_error("Invalid prepared CPU region extent or source index");
        for (const auto i : indices)
            if (i < 0 || static_cast<size_t>(i) >= sources_.size())
                throw py::value_error("Invalid prepared CPU region extent or source index");
    }
    size_t gather(const std::vector<int64_t> &indices, size_t offset, size_t length, py::buffer destination,
                  bool parallel) {
        check(indices);
        if (!length || offset > record_bytes_ || length > record_bytes_ - offset)
            throw py::value_error("Invalid prepared CPU region extent or source index");
        auto out = destination.request(true);
        if (out.readonly || out.ndim != 1 || out.itemsize != 1 || out.strides[0] != 1 ||
            static_cast<size_t>(out.size) != indices.size() * length)
            throw std::invalid_argument("Prepared CPU region destination has a different extent");
        char *base = static_cast<char *>(out.ptr);
        std::vector<const char *> from(indices.size());
        for (size_t k = 0; k < indices.size(); ++k)
            from[k] = static_cast<const char *>(sources_[indices[k]].ptr) + offset;
        {
            // Serial by default, like the original inline copy: parallel layer
            // gathers during the forward slow the forward itself (m4nfg). The
            // caller requests parallel copies only while the GPU is idle at the
            // layer-zero guard of a promised refresh.
            py::gil_scoped_release release;
            if (parallel)
                return kvmem_parallel::for_rows(indices.size(), [&](size_t k) {
                    std::memcpy(base + k * length, from[k], length);
                });
            for (size_t k = 0; k < indices.size(); ++k)
                std::memcpy(base + k * length, from[k], length);
        }
        return 1;
    }
private:
    size_t record_bytes_;
    std::vector<py::buffer_info> sources_;
};

// Opt-in streaming publication (streaming_publication.py): one session's
// record pages, filled one model layer at a time while the forward runs.
// Built once with the GIL; each layer() call validates nothing new and
// releases the GIL before any work, so the engine thread is never starved by
// per-layer Python job lists. Copies are plain memcpy of contiguous token rows
// and the MeanK sums use exactly the finalize_mean_batch arithmetic (SIMD when
// eligible, else the scalar token-order FP32 loop), one layer at a time.
class StreamPages {
public:
    StreamPages(py::buffer source_raw, py::buffer source_values,
                const std::vector<py::buffer> &page_raw, const std::vector<py::buffer> &page_values,
                const std::vector<py::buffer> &page_sums, ssize_t block)
        : block_(block) {
        source_raw_ = source_raw.request();
        source_values_ = source_values.request();
        const auto &s = source_raw_;
        if (s.ndim != 4 || s.itemsize != 1 || source_values_.ndim != 4 || source_values_.itemsize != 1 ||
            s.shape != source_values_.shape || block <= 0 || page_raw.empty() ||
            page_raw.size() != page_values.size() || page_raw.size() != page_sums.size() ||
            s.shape[1] != static_cast<ssize_t>(page_raw.size()) * block ||
            !rows_contiguous(s) || !rows_contiguous(source_values_) ||
            source_values_.strides != s.strides)
            throw std::invalid_argument("Invalid streamed page source");
        layers_ = s.shape[0];
        row_bytes_ = s.shape[2] * s.shape[3];
        for (size_t i = 0; i < page_raw.size(); ++i) {
            auto raw = page_raw[i].request(true);
            auto values = page_values[i].request(true);
            auto sums = page_sums[i].request(true);
            if (raw.ndim != 4 || raw.itemsize != 1 || raw.readonly || values.readonly ||
                raw.shape != std::vector<ssize_t>{layers_, block, s.shape[2], s.shape[3]} ||
                values.shape != raw.shape || values.itemsize != 1 || !contiguous(raw) || !contiguous(values) ||
                sums.ndim != 3 || sums.itemsize != 4 || sums.readonly ||
                sums.format != py::format_descriptor<float>::format() ||
                sums.shape[0] <= 0 || sums.shape[0] > layers_ || sums.shape[1] != s.shape[2] ||
                sums.shape[2] != s.shape[3] || !contiguous(sums))
                throw std::invalid_argument("Invalid streamed record page");
            if (i == 0) scoring_ = sums.shape[0];
            else if (sums.shape[0] != scoring_)
                throw std::invalid_argument("Streamed pages disagree on scoring layers");
            raw_.push_back(std::move(raw));
            values_.push_back(std::move(values));
            sums_.push_back(std::move(sums));
        }
        std::vector<const py::buffer_info *> written, read{&source_raw_, &source_values_};
        for (size_t i = 0; i < raw_.size(); ++i) {
            written.push_back(&raw_[i]);
            written.push_back(&values_[i]);
            written.push_back(&sums_[i]);
        }
        if (!disjoint_buffers(written, read))
            throw std::invalid_argument("Streamed pages overlap each other or their source");
    }

    // Returns how many pages took the vectorized MeanK path for this layer.
    size_t layer(ssize_t layer, bool vectorized) {
        if (layer < 0 || layer >= layers_) throw std::out_of_range("Streamed layer outside the session");
        std::vector<uint8_t> simd(raw_.size(), 0);
        {
            py::gil_scoped_release release;
            const size_t bytes = static_cast<size_t>(block_ * row_bytes_);
            const char *src_raw = static_cast<const char *>(source_raw_.ptr) + layer * source_raw_.strides[0];
            const char *src_values = static_cast<const char *>(source_values_.ptr) + layer * source_values_.strides[0];
            auto fill = [&](size_t page) {
                char *raw = static_cast<char *>(raw_[page].ptr) + layer * raw_[page].strides[0];
                char *values = static_cast<char *>(values_[page].ptr) + layer * values_[page].strides[0];
                std::memcpy(raw, src_raw + page * bytes, bytes);
                std::memcpy(values, src_values + page * bytes, bytes);
                if (layer >= scoring_) return;
                // Single-layer views: identical to finalize_mean_batch on raw[layer:layer+1].
                const auto &r = raw_[page];
                const auto &m = sums_[page];
                py::buffer_info a(raw, 1, "B", 4, {ssize_t(1), r.shape[1], r.shape[2], r.shape[3]},
                                  {r.strides[0], r.strides[1], r.strides[2], r.strides[3]});
                py::buffer_info b(static_cast<char *>(m.ptr) + layer * m.strides[0], 4,
                                  py::format_descriptor<float>::format(), 3,
                                  {ssize_t(1), m.shape[1], m.shape[2]}, {m.strides[0], m.strides[1], m.strides[2]});
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
                if (vectorized && mean_fp8_simd_eligible(a, b, block_)) {
                    finalize_mean_fp8_simd(a, b, block_);
                    simd[page] = 1;
                    return;
                }
#endif
                auto *out = static_cast<float *>(b.ptr);
                const ssize_t channels = b.shape[1] * b.shape[2];
                std::fill(out, out + channels, 0.f);
                for (ssize_t token = 0; token < block_; ++token) {
                    const char *row = raw + token * a.strides[1];
                    for (ssize_t channel = 0; channel < channels; ++channel)
                        out[channel] += fp8_as_float(row + channel);
                }
            };
            kvmem_parallel::for_rows(raw_.size(), fill);
        }
        size_t total = 0;
        for (auto flag : simd) total += flag;
        return total;
    }

    ssize_t layers() const { return layers_; }
    ssize_t scoring() const { return scoring_; }

private:
    // Token rows of each layer are contiguous; layers may be strided (a slice
    // of a larger reusable pinned staging buffer).
    static bool rows_contiguous(const py::buffer_info &v) {
        return v.ndim == 4 && v.strides[3] == 1 && v.strides[2] == v.shape[3] &&
               v.strides[1] == v.shape[2] * v.shape[3] && v.strides[0] >= v.shape[1] * v.strides[1];
    }
    static bool contiguous(const py::buffer_info &v) {
        ssize_t expected = v.itemsize;
        for (ssize_t axis = v.ndim - 1; axis >= 0; --axis) {
            if (v.strides[axis] != expected) return false;
            expected *= v.shape[axis];
        }
        return true;
    }
    py::buffer_info source_raw_, source_values_;
    std::vector<py::buffer_info> raw_, values_, sums_;
    ssize_t block_ = 0, layers_ = 0, row_bytes_ = 0, scoring_ = 0;
};

// KVMEM_SESSION_FAST_TURN: (slot, generation) tuples to handles. False unless
// the input is an exact list of exact 2-tuples of exact ints in range.
static bool handles_from_pairs(const py::object &pairs, std::vector<KvMemBlockHandle> &out) {
    if (!PyList_CheckExact(pairs.ptr())) return false;
    const Py_ssize_t count = PyList_GET_SIZE(pairs.ptr());
    out.reserve(static_cast<size_t>(count));
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *row = PyList_GET_ITEM(pairs.ptr(), i);
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2) return false;
        PyObject *slot = PyTuple_GET_ITEM(row, 0);
        PyObject *generation = PyTuple_GET_ITEM(row, 1);
        if (!PyLong_CheckExact(slot) || !PyLong_CheckExact(generation)) return false;
        int overflow = 0;
        const long long s = PyLong_AsLongLongAndOverflow(slot, &overflow);
        if (overflow || s < 0 || s > static_cast<long long>(UINT32_MAX)) return false;
        const unsigned long long g = PyLong_AsUnsignedLongLong(generation);
        if (g == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        KvMemBlockHandle handle;
        handle.slot = static_cast<uint32_t>(s);
        handle.generation = static_cast<uint64_t>(g);
        out.push_back(handle);
    }
    return true;
}

PYBIND11_MODULE(_native, m) {
    bind_historical_thinking(m);
    m.def("checkpoint_token_bytes", [](py::object tokens, Py_ssize_t count) {
        // Match token_bytes' exact builtin-int and uint32 validation. Do not
        // accept bool, int subclasses, truncating casts, or hash-only matches.
        const bool tuple = PyTuple_CheckExact(tokens.ptr());
        if (!tuple && !PyList_CheckExact(tokens.ptr()))
            throw py::value_error("Native token identity requires a list or tuple");
        const auto size = tuple ? PyTuple_GET_SIZE(tokens.ptr()) : PyList_GET_SIZE(tokens.ptr());
        if (count <= 0 || count >= size)
            throw py::value_error("Prompt checkpoint must leave at least one token for native prefill");
        if (count > PY_SSIZE_T_MAX / 4)
            throw py::value_error("Token identity size overflow");
        py::bytes result = py::reinterpret_steal<py::bytes>(PyBytes_FromStringAndSize(nullptr, count * 4));
        if (!result) throw py::error_already_set();
        char *destination = PyBytes_AS_STRING(result.ptr());
        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject *token = tuple ? PyTuple_GET_ITEM(tokens.ptr(), i) : PyList_GET_ITEM(tokens.ptr(), i);
            if (!PyLong_CheckExact(token))
                throw py::value_error("Prompt identity requires exact unsigned token IDs");
            const auto value = PyLong_AsUnsignedLongLong(token);
            if (PyErr_Occurred() || value > UINT32_MAX) {
                PyErr_Clear();
                throw py::value_error("Prompt identity requires exact unsigned token IDs");
            }
            const uint32_t packed = static_cast<uint32_t>(value);
            std::memcpy(destination + i * 4, &packed, 4);
        }
        return result;
    });
    kvmem_fps_cpu::bind(m);
    py::register_exception<KvMemAdmissionRetry>(m, "HistoryAdmissionRetry");
    kvmem_prefill::bind(m);
    kvmem_graph::bind(m);
    kvmem_decode::bind(m);
    kvmem_propagation::bind(m);
    kvmem_download::bind(m);
    using ActionParser = detail::ActionAttentionParser;
    py::class_<ActionParser>(m, "ActionParser")
        .def(py::init([](std::string scope, uint32_t interval, const std::vector<int32_t> &ids, bool thinking) {
            if (ids.size()!=4 || !interval || interval>4096)
                throw std::invalid_argument("Invalid action parser settings");
            ActionParser p;
            p.scope=ActionParser::parse_scope(scope); p.interval=interval;
            p.open=ids[0]; p.close=ids[1]; p.think_open=ids[2]; p.think_close=ids[3];
            p.committed.thinking=thinking;
            return p;
        }))
        .def("prepare", [](ActionParser &p, const std::vector<uint32_t> &tokens) {
            if (tokens.size()>64) throw std::invalid_argument("Action verifier exceeds 64 rows");
            p.prepare(tokens.data(), tokens.size()); return p.mask;
        })
        .def("commit", &ActionParser::commit)
        .def_property_readonly("state", [](const ActionParser &p) {
            return py::make_tuple(p.committed.thinking,p.committed.action,p.committed.tokens);
        });
    m.def("validate_action_options", [](std::string scope, std::string layers, uint32_t interval, double decay) {
        KvMemAttentionConfig c; c.scope=scope; c.layers=layers; c.interval=interval; c.decay=decay; c.validate();
    });
    m.def("resolve_action_layers", &detail::resolve_attention_layers);
    using QueryHistory = detail::KvMemQueryHistory;
    py::class_<QueryHistory::Snapshot>(m, "QueryHistorySnapshot");
    py::class_<QueryHistory>(m, "QueryHistory")
        .def(py::init<>())
        .def("begin", &QueryHistory::begin)
        .def("preview", &QueryHistory::preview)
        .def("commit", &QueryHistory::commit)
        .def("abort", &QueryHistory::abort)
        .def("snapshot", &QueryHistory::snapshot)
        .def("restore", &QueryHistory::restore)
        .def_property_readonly("scores", &QueryHistory::scores)
        .def_property_readonly("committed_scores", &QueryHistory::committed_scores)
        .def_property_readonly("open", &QueryHistory::open)
        .def_property_readonly("has_observation", &QueryHistory::has_observation)
        .def_property_readonly("input_end", &QueryHistory::input_end)
        .def_property_readonly("round_count", &QueryHistory::round_count);
    m.def("fit_mandatory_blocks", [](uint32_t prompt, uint32_t budget, uint32_t sink, uint32_t recent,
            const std::vector<std::array<uint32_t,3>> &spans, uint32_t live_begin, uint32_t live_end,
            bool pin_live) {
        if (!budget || prompt > (1u<<24)) throw std::invalid_argument("Invalid mandatory capacity");
        detail::KvmemMandatoryFitInput input;
        input.prompt_blocks=prompt; input.budget_blocks=budget;
        input.sink_blocks=sink; input.recent_blocks=recent;
        input.live_begin=live_begin; input.live_end=live_end; input.pin_live_suffix=pin_live;
        for (const auto &span:spans) {
            if (span[0]>=span[1] || span[2]>3) throw std::invalid_argument("Invalid mandatory span");
            input.hard_spans.push_back({span[0],span[1],static_cast<uint8_t>(span[2])});
        }
        const auto result=detail::kvmem_fit_mandatory_blocks(input);
        py::dict out;
        out["hard_blocks"]=result.hard_blocks; out["raw_mandatory_blocks"]=result.raw_mandatory_blocks;
        out["fitted_live_blocks"]=result.fitted_live_blocks; out["soft_retrievable_blocks"]=result.soft_retrievable_blocks;
        out["retrieval_reserve_blocks"]=result.retrieval_reserve_blocks;
        out["capacity_fitted"]=result.capacity_fitted; out["replay_begin_block"]=result.replay_begin_block;
        return out;
    }, py::arg("prompt_blocks"),py::arg("budget_blocks"),py::arg("sink_blocks"),py::arg("recent_blocks"),
       py::arg("spans"),py::arg("live_begin")=0,py::arg("live_end")=0,py::arg("pin_live")=false);
    m.def("replay_capture_begin", &detail::kvmem_replay_capture_begin);
    m.def("branch_safe_checkpoint", &detail::kvmem_branch_safe_prompt_checkpoint_split);
    m.def("finalize_mean_bf16", [](py::buffer raw, py::buffer sums, ssize_t count) {
        auto a=raw.request(), b=sums.request(true);
        if (a.ndim!=4 || a.itemsize!=1 || b.ndim!=3 || b.itemsize!=4 ||
            b.format!=py::format_descriptor<float>::format() || count<=0 || count>a.shape[1] ||
            b.shape[0]>a.shape[0] || b.shape[1]!=a.shape[2] || b.shape[2]*2!=a.shape[3] ||
            b.shape[0]<=0 || b.shape[1]<=0 || b.shape[2]<=0 ||
            a.strides[3]!=1 || a.strides[2]!=a.shape[3] ||
            a.strides[1]<a.shape[2]*a.shape[3] || a.strides[0]<a.shape[1]*a.strides[1] || b.strides[2]!=4 ||
            b.strides[1]!=b.shape[2]*4 || b.strides[0]!=b.shape[1]*b.strides[1])
            throw std::invalid_argument("Invalid BF16 content-index buffers");
        py::gil_scoped_release release;
        for (ssize_t l=0;l<b.shape[0];++l) {
            float *out=reinterpret_cast<float*>(static_cast<char*>(b.ptr)+l*b.strides[0]);
            const ssize_t channels=b.shape[1]*b.shape[2];
            std::fill(out,out+channels,0.f);
            for (ssize_t t=0;t<count;++t) {
                const char *row=static_cast<const char*>(a.ptr)+l*a.strides[0]+t*a.strides[1];
                for (ssize_t d=0;d<channels;++d) out[d]+=bf16_as_float(row+d*2);
            }
        }
    });
    m.def("finalize_mean_fp8", [](py::buffer raw, py::buffer sums, ssize_t count) {
        auto a=raw.request(), b=sums.request(true);
        if (a.ndim!=4 || a.itemsize!=1 || b.ndim!=3 || b.itemsize!=4 || b.readonly ||
            b.format!=py::format_descriptor<float>::format() || count<=0 || count>a.shape[1] ||
            b.shape[0]<=0 || b.shape[0]>a.shape[0] || b.shape[1]!=a.shape[2] || b.shape[2]!=a.shape[3] ||
            b.shape[1]<=0 || b.shape[2]<=0 || a.strides[3]!=1 || a.strides[2]!=a.shape[3] ||
            a.strides[1]<a.shape[2]*a.shape[3] || a.strides[0]<a.shape[1]*a.strides[1] ||
            b.strides[2]!=4 || b.strides[1]!=b.shape[2]*4 || b.strides[0]!=b.shape[1]*b.strides[1])
            throw std::invalid_argument("Invalid FP8 content-index buffers");
        py::gil_scoped_release release;
        for (ssize_t layer=0;layer<b.shape[0];++layer) {
            float *out=reinterpret_cast<float*>(static_cast<char*>(b.ptr)+layer*b.strides[0]);
            const ssize_t channels=b.shape[1]*b.shape[2];
            std::fill(out,out+channels,0.f);
            for (ssize_t token=0;token<count;++token) {
                const char *row=static_cast<const char*>(a.ptr)+layer*a.strides[0]+token*a.strides[1];
                for (ssize_t channel=0;channel<channels;++channel) out[channel]+=fp8_as_float(row+channel);
            }
        }
    });
    m.def("finalize_mean_simd_supported", &native_decode_sum_supported);
    m.def("graph_size", [](py::handle value, py::handle seen) {
        if (!PySet_Check(seen.ptr())) throw py::type_error("graph_size needs a set of seen ids");
        py::object fields = py::module_::import("dataclasses").attr("fields");
        return graph_size_impl(value.ptr(), seen.ptr(), fields.ptr());
    }, py::arg("value"), py::arg("seen"));
    py::class_<LeaseSet>(m, "LeaseSet")
        .def("valid", &LeaseSet::valid)
        .def("reset", &LeaseSet::reset)
        .def("clear", &LeaseSet::reset)
        .def("__len__", [](const LeaseSet &s) { return s.leases.size(); });
    py::class_<RecordSources>(m, "RecordSources")
        .def(py::init<const std::vector<py::buffer> &, size_t>(), py::arg("sources"), py::arg("record_bytes"))
        .def("__len__", &RecordSources::size)
        .def("check", &RecordSources::check, py::arg("indices"))
        .def("extend", &RecordSources::extend, py::arg("sources"))
        .def("gather", &RecordSources::gather, py::arg("indices"), py::arg("offset"), py::arg("length"),
             py::arg("destination"), py::arg("parallel") = false);
    // PressureReader.read's per-record validation, in its original `or` order:
    // [entry[3] for each record] or ValueError with the original message.
    m.def("prepared_record_indices", [](const py::iterable &records, const py::dict &table, const py::object &block) {
        py::list out;
        auto fail = []() { throw py::value_error("Prepared pressure source is absent, partial or changed"); };
        for (const auto &record : records) {
            py::object key = py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(record.ptr()));
            if (!key) throw py::error_already_set();
            PyObject *entry = PyDict_GetItemWithError(table.ptr(), key.ptr());
            if (!entry) { if (PyErr_Occurred()) throw py::error_already_set(); fail(); }
            if (!PyTuple_Check(entry) || PyTuple_GET_SIZE(entry) < 4) fail();
            if (PyTuple_GET_ITEM(entry, 0) != record.ptr()) fail();
            if (record.attr("handle").ptr() != PyTuple_GET_ITEM(entry, 1)) fail();
            const int begin_differs = PyObject_RichCompareBool(record.attr("history_begin").ptr(),
                                                               PyTuple_GET_ITEM(entry, 2), Py_NE);
            if (begin_differs < 0) throw py::error_already_set();
            if (begin_differs) fail();
            const int count_differs = PyObject_RichCompareBool(record.attr("count").ptr(), block.ptr(), Py_NE);
            if (count_differs < 0) throw py::error_already_set();
            if (count_differs) fail();
            const int archived = PyObject_IsTrue(record.attr("archived").ptr());
            if (archived < 0) throw py::error_already_set();
            if (!archived) fail();
            out.append(py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(entry, 3)));
        }
        return out;
    }, py::arg("records"), py::arg("table"), py::arg("block"));
    // PressureReader._collect_batched's Python work: first-visit unique records
    // across layers, count == block and archived, handles, (slot, generation)
    // pairs, alias/foreign checks and the id -> (record, handle, begin, i) table.
    m.def("collect_pressure_sources", [](const py::iterable &layers, const py::object &block,
                                         const py::dict &authority_records) {
        std::unordered_set<PyObject *> visited;
        std::vector<py::object> unique;
        for (const auto &layer : layers)
            for (const auto &record : py::reinterpret_borrow<py::iterable>(layer))
                if (visited.insert(record.ptr()).second) unique.push_back(py::reinterpret_borrow<py::object>(record));
        py::list records, handles, pairs;
        py::dict table;
        std::set<std::pair<uint32_t, uint64_t>> keys;
        bool aliased = false;
        for (const auto &record : unique) {
            const int full = PyObject_RichCompareBool(record.attr("count").ptr(), block.ptr(), Py_EQ);
            if (full < 0) throw py::error_already_set();
            if (!full) continue;
            const int archived = PyObject_IsTrue(record.attr("archived").ptr());
            if (archived < 0) throw py::error_already_set();
            if (!archived) continue;
            records.append(record);
        }
        for (size_t i = 0; i < records.size(); ++i) {
            py::object record = records[i];
            py::object handle = record.attr("handle");
            const auto value = py::cast<KvMemBlockHandle>(handle);
            handles.append(handle);
            pairs.append(py::make_tuple(value.slot, value.generation));
            if (!keys.insert({value.slot, value.generation}).second) aliased = true;
        }
        if (!aliased) {
            for (size_t i = 0; i < records.size(); ++i) {
                PyObject *found = PyDict_GetItemWithError(authority_records.ptr(), py::object(pairs[i]).ptr());
                if (!found && PyErr_Occurred()) throw py::error_already_set();
                if (found != py::object(records[i]).ptr()) { aliased = true; break; }
            }
        }
        if (aliased) throw py::value_error("Pressure sources have aliased or foreign generations");
        for (size_t i = 0; i < records.size(); ++i) {
            py::object record = records[i];
            py::object key = py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(record.ptr()));
            table[key] = py::make_tuple(record, handles[i], record.attr("history_begin"), i);
        }
        return py::make_tuple(records, handles, pairs, table);
    }, py::arg("layers"), py::arg("block"), py::arg("authority_records"));
    // Opt-in KVMEM_LAYER0_FAST: KvMemStore::pick_prefill_pressure_from in
    // history-position space, for a repository whose live blocks are exactly
    // the aligned history [0, history_end) (no GC; every block live). The
    // layer-0 fast path picks before the chunk's records have slots; the
    // original store pick still runs afterwards and must agree (checked).
    // Same steps: sink prefix, mandatory, positive candidates by score with
    // later-position tie break, then the most recent remaining candidates.
    m.def("pressure_pick_positions", [](int64_t history_end, int64_t block_tokens, int64_t sink_blocks,
            int64_t budget_blocks_limit, const std::vector<int64_t> &mandatory,
            const std::vector<int64_t> &candidates, const std::vector<double> &scores,
            int64_t budget_tokens) {
        if (block_tokens <= 0 || history_end < 0 || history_end % block_tokens || sink_blocks < 0)
            throw std::invalid_argument("Position pick requires an aligned history");
        if (candidates.size() != scores.size())
            throw std::invalid_argument("Position pick candidates and scores differ in length");
        if (budget_tokens && (budget_tokens % block_tokens || budget_tokens / block_tokens > budget_blocks_limit ||
                budget_tokens / block_tokens < sink_blocks))
            throw std::invalid_argument("Invalid bounded pressure budget");
        const int64_t n = history_end / block_tokens;
        std::vector<int64_t> selected;
        if (n == 0) return selected;
        auto index_of = [&](int64_t begin) -> int64_t {
            if (begin < 0 || begin % block_tokens || begin >= history_end)
                throw std::runtime_error("KVMem mandatory block is outside the pressure snapshot");
            return begin / block_tokens;
        };
        for (const auto begin : mandatory) (void)index_of(begin);
        const int64_t budget = budget_tokens ? budget_tokens / block_tokens : budget_blocks_limit;
        std::vector<uint8_t> resident(static_cast<size_t>(n), 0);
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (!std::isfinite(scores[i]) || scores[i] < 0)
                throw std::invalid_argument("invalid pressure action score");
            const int64_t begin = candidates[i];
            if (begin >= 0 && begin % block_tokens == 0 && begin < history_end)
                resident[static_cast<size_t>(begin / block_tokens)] = 1;
        }
        if (budget == 0 || n <= budget) {
            for (const auto begin : mandatory) resident[static_cast<size_t>(begin / block_tokens)] = 1;
            for (int64_t i = 0; i < n; ++i)
                if (candidates.empty() || resident[static_cast<size_t>(i)] || i < sink_blocks)
                    selected.push_back(i * block_tokens);
            return selected;
        }
        std::vector<uint8_t> kept(static_cast<size_t>(n), 0);
        int64_t count = 0;
        auto keep = [&](int64_t i) { if (!kept[static_cast<size_t>(i)]) { kept[static_cast<size_t>(i)] = 1; ++count; } };
        // fixed_selection_from(order, mandatory, pressure=true, budget)
        {
            std::vector<uint8_t> fixed(static_cast<size_t>(n), 0);
            int64_t fixed_count = 0;
            auto fix = [&](int64_t i) { if (!fixed[static_cast<size_t>(i)]) { fixed[static_cast<size_t>(i)] = 1; ++fixed_count; } };
            for (int64_t i = 0; i < n; ++i) if (i < sink_blocks && fixed_count < budget) fix(i);
            for (const auto begin : mandatory) fix(begin / block_tokens);
            if (fixed_count > budget)
                throw std::runtime_error("KVMem mandatory selection plus sink blocks exceeds the configured selection budget");
            for (int64_t i = 0; i < n; ++i) if (fixed[static_cast<size_t>(i)]) keep(i);
        }
        std::vector<std::pair<int64_t, double>> ranked;
        ranked.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            const int64_t begin = candidates[i];
            if (begin < 0 || begin % block_tokens || begin >= history_end) continue;
            const int64_t index = begin / block_tokens;
            if (!kept[static_cast<size_t>(index)] && scores[i] > 0) ranked.emplace_back(index, scores[i]);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            return a.second != b.second ? a.second > b.second : b.first < a.first;
        });
        for (const auto &[index, score] : ranked) {
            (void)score;
            if (count >= budget) break;
            keep(index);
        }
        for (int64_t i = n - 1; i >= 0 && count < budget; --i) {
            if (!resident[static_cast<size_t>(i)]) continue;
            keep(i);
        }
        for (int64_t i = 0; i < n; ++i) if (kept[static_cast<size_t>(i)]) selected.push_back(i * block_tokens);
        return selected;
    }, py::arg("history_end"), py::arg("block_tokens"), py::arg("sink_blocks"), py::arg("budget_blocks_limit"),
       py::arg("mandatory"), py::arg("candidates"), py::arg("scores"), py::arg("budget_tokens"));
    // MemoryArchive.prepare_region_reader's entry checks and _region_sources,
    // with the original exceptions (call under the archive and tier locks).
    m.def("archive_region_sources", [](const py::dict &entries, const py::dict &tier_records,
                                       const py::list &keys, size_t record_bytes) {
        std::vector<PyObject *> found;
        found.reserve(keys.size());
        bool missing = false;
        for (const auto &key : keys) {
            PyObject *entry = PyDict_GetItemWithError(entries.ptr(), key.ptr());
            if (!entry && PyErr_Occurred()) throw py::error_already_set();
            found.push_back(entry);
        }
        for (PyObject *entry : found) {
            if (!entry) { missing = true; break; }
            const int ready = PyObject_IsTrue(py::handle(entry).attr("ready").ptr());
            if (ready < 0) throw py::error_already_set();
            if (!ready) { missing = true; break; }
        }
        if (missing) {
            PyErr_SetObject(PyExc_KeyError, py::str("Prepared CPU record is absent, stale or unpublished").ptr());
            throw py::error_already_set();
        }
        py::list sources;
        for (PyObject *entry : found) {
            py::object slot = py::handle(entry).attr("allocation").attr("slot");
            PyObject *source = PyDict_GetItemWithError(tier_records.ptr(), slot.ptr());
            if (!source) {
                if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, slot.ptr());
                throw py::error_already_set();
            }
            sources.append(py::handle(source));
        }
        for (const auto &source : sources)
            if (!PyBytes_CheckExact(source.ptr()) || static_cast<size_t>(PyBytes_GET_SIZE(source.ptr())) != record_bytes)
                throw py::value_error("Prepared CPU regions require exact immutable record bytes");
        return sources;
    }, py::arg("entries"), py::arg("tier_records"), py::arg("keys"), py::arg("record_bytes"));
    // pressure_commit.prepare_commit's per-layer map: entries = [identities[s]],
    // the same length and (history_begin, count) checks in the same order, and
    // rows (*entry, offset) with offsets accumulated from 0 by Python addition.
    m.def("commit_map", [](const py::list &slots, const py::dict &identities, const py::list &picked) {
        std::vector<py::object> entries;
        entries.reserve(slots.size());
        for (const auto &slot : slots) {
            PyObject *entry = PyDict_GetItemWithError(identities.ptr(), slot.ptr());
            if (!entry) {
                if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, slot.ptr());
                throw py::error_already_set();
            }
            entries.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        auto fail = []() { throw std::runtime_error("Prepared pressure map differs from the selected record order"); };
        if (entries.size() != picked.size()) fail();
        for (size_t i = 0; i < entries.size(); ++i) {
            py::tuple entry = py::reinterpret_borrow<py::tuple>(entries[i]);
            py::handle record = picked[i];
            const int begin = PyObject_RichCompareBool(entry[2].ptr(), record.attr("history_begin").ptr(), Py_NE);
            if (begin < 0) throw py::error_already_set();
            if (begin) fail();
            const int count = PyObject_RichCompareBool(entry[3].ptr(), record.attr("count").ptr(), Py_NE);
            if (count < 0) throw py::error_already_set();
            if (count) fail();
        }
        py::list rows;
        py::object offset = py::int_(0);
        for (const auto &item : entries) {
            py::tuple entry = py::reinterpret_borrow<py::tuple>(item);
            rows.append(py::make_tuple(entry[0], entry[1], entry[2], entry[3], offset));
            PyObject *next = PyNumber_Add(offset.ptr(), entry[3].ptr());
            if (!next) throw py::error_already_set();
            offset = py::reinterpret_steal<py::object>(next);
        }
        return py::make_tuple(py::tuple(rows), offset);
    }, py::arg("slots"), py::arg("identities"), py::arg("picked"));
    // pressure_resident_values.resident_bases: the same map checks and messages
    // in the same order; the old dense base of every chosen block, or None at
    // the first block that is not a whole archived block of the dense window.
    // CPython API on the borrowed rows/records (no per-item pybind wrappers).
    m.def("reuse_resident_bases", [](const py::object &mapping, const py::object &chosen,
                                     int64_t local_tokens, int64_t block_tokens) -> py::object {
        auto as_i64 = [](PyObject *value) {
            const long long v = PyLong_AsLongLong(value);
            if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
            return static_cast<int64_t>(v);
        };
        py::object rows_fast = py::reinterpret_steal<py::object>(
            PySequence_Fast(mapping.ptr(), "Resident pressure source map must be a sequence"));
        if (!rows_fast) throw py::error_already_set();
        const Py_ssize_t rows = PySequence_Fast_GET_SIZE(rows_fast.ptr());
        PyObject **row_items = PySequence_Fast_ITEMS(rows_fast.ptr());
        std::unordered_map<int64_t, std::pair<int64_t, int64_t>> by_begin;
        by_begin.reserve(static_cast<size_t>(rows) * 2 + 1);
        int64_t cursor = 0;
        for (Py_ssize_t i = 0; i < rows; ++i) {
            PyObject *row = row_items[i];
            if (!PyTuple_Check(row) || PyTuple_GET_SIZE(row) != 5)
                throw py::value_error("Resident pressure source map rows must have five fields");
            const auto begin = as_i64(PyTuple_GET_ITEM(row, 2)), count = as_i64(PyTuple_GET_ITEM(row, 3));
            const auto position = as_i64(PyTuple_GET_ITEM(row, 4));
            if (position != cursor || count <= 0 || count > block_tokens || by_begin.count(begin) ||
                    begin < 0 || position + count > local_tokens)
                throw py::value_error("Resident pressure source map is not a unique dense accepted window");
            by_begin.emplace(begin, std::make_pair(count, position));
            cursor += count;
        }
        if (cursor != local_tokens)
            throw py::value_error("Resident pressure map does not cover the accepted native window");
        py::object chosen_fast = py::reinterpret_steal<py::object>(
            PySequence_Fast(chosen.ptr(), "Chosen pressure records must be a sequence"));
        if (!chosen_fast) throw py::error_already_set();
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(chosen_fast.ptr());
        PyObject **records = PySequence_Fast_ITEMS(chosen_fast.ptr());
        py::str begin_name("history_begin"), count_name("count"), archived_name("archived");
        auto attr_i64 = [&](PyObject *record, const py::str &name) {
            py::object value = py::reinterpret_steal<py::object>(PyObject_GetAttr(record, name.ptr()));
            if (!value) throw py::error_already_set();
            return as_i64(value.ptr());
        };
        std::vector<int64_t> begins(static_cast<size_t>(n));
        std::unordered_set<int64_t> seen;
        seen.reserve(static_cast<size_t>(n) * 2 + 1);
        for (Py_ssize_t i = 0; i < n; ++i) {
            begins[i] = attr_i64(records[i], begin_name);
            seen.insert(begins[i]);
        }
        if (static_cast<Py_ssize_t>(seen.size()) != n) throw py::value_error("Repeated selected pressure source");
        py::list bases(n);
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (attr_i64(records[i], count_name) != block_tokens) return py::none();
            py::object archived = py::reinterpret_steal<py::object>(PyObject_GetAttr(records[i], archived_name.ptr()));
            if (!archived) throw py::error_already_set();
            const int truth = PyObject_IsTrue(archived.ptr());
            if (truth < 0) throw py::error_already_set();
            if (!truth) return py::none();
            const auto source = by_begin.find(begins[i]);
            if (source == by_begin.end() || source->second.first != block_tokens ||
                    source->second.second % block_tokens)
                return py::none();
            PyObject *base = PyLong_FromLongLong(source->second.second);
            if (!base) throw py::error_already_set();
            PyList_SET_ITEM(bases.ptr(), i, base);
        }
        return bases;
    }, py::arg("mapping"), py::arg("chosen"), py::arg("local_tokens"), py::arg("block_tokens"));
    // pressure_key_reuse.plan_layer over chosen records plus assemble_reused's
    // metadata rows: V sources/destinations of every moved block, then reused K
    // sources, destinations, deltas, payload indices and modes (REUSE = 0).
    // Returns (metadata int64 array, moved, reused, [(index, destination)] of
    // reloads, new_counts, skipped); same checks and messages as plan_layer.
    m.def("reuse_layer_plan", [](const py::object &chosen, const py::object &sources,
                                 const py::dict &counts, int64_t limit, bool untrusted) {
        auto as_i64 = [](PyObject *value) {
            const long long v = PyLong_AsLongLong(value);
            if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
            return static_cast<int64_t>(v);
        };
        py::object chosen_fast = py::reinterpret_steal<py::object>(
            PySequence_Fast(chosen.ptr(), "Chosen pressure records must be a sequence"));
        if (!chosen_fast) throw py::error_already_set();
        py::object sources_fast = py::reinterpret_steal<py::object>(
            PySequence_Fast(sources.ptr(), "Reuse sources must be a sequence"));
        if (!sources_fast) throw py::error_already_set();
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(chosen_fast.ptr());
        if (PySequence_Fast_GET_SIZE(sources_fast.ptr()) != n)
            throw py::value_error("Reuse plan needs one source per kept block");
        PyObject **records = PySequence_Fast_ITEMS(chosen_fast.ptr());
        PyObject **source_items = PySequence_Fast_ITEMS(sources_fast.ptr());
        py::str begin_name("history_begin"), count_name("count");
        std::vector<int64_t> v_src, v_dst, k_src, k_dst, k_delta;
        v_src.reserve(n); v_dst.reserve(n); k_src.reserve(n); k_dst.reserve(n); k_delta.reserve(n);
        py::list reload;
        py::dict new_counts;
        int64_t cursor = 0, previous = 0, skipped = 0;
        for (Py_ssize_t i = 0; i < n; ++i) {
            py::object begin = py::reinterpret_steal<py::object>(PyObject_GetAttr(records[i], begin_name.ptr()));
            if (!begin) throw py::error_already_set();
            py::object tokens = py::reinterpret_steal<py::object>(PyObject_GetAttr(records[i], count_name.ptr()));
            if (!tokens) throw py::error_already_set();
            const int64_t source = as_i64(source_items[i]);
            const int64_t destination = cursor;
            cursor += as_i64(tokens.ptr());
            const int64_t delta = source - destination;
            if (delta < 0 || delta < previous)
                throw py::value_error("Pressure compaction must shift kept blocks left, non-decreasingly");
            previous = delta;
            int64_t count = limit;
            if (!untrusted) {
                PyObject *found = PyDict_GetItemWithError(counts.ptr(), begin.ptr());
                if (!found && PyErr_Occurred()) throw py::error_already_set();
                count = found ? as_i64(found) : 0;
            }
            int64_t next_count = count;
            if (delta != 0) {
                v_src.push_back(source);
                v_dst.push_back(destination);
                if (count < limit) {
                    next_count = count + 1;
                    k_src.push_back(source);
                    k_dst.push_back(destination);
                    k_delta.push_back(delta);
                } else {
                    next_count = 0;
                    reload.append(py::make_tuple(i, destination));
                }
            } else {
                ++skipped;
            }
            py::object value = py::reinterpret_steal<py::object>(PyLong_FromLongLong(next_count));
            if (!value || PyDict_SetItem(new_counts.ptr(), begin.ptr(), value.ptr()) < 0)
                throw py::error_already_set();
        }
        const size_t m = v_src.size(), r = k_src.size();
        py::array_t<int64_t> meta(static_cast<py::ssize_t>(2 * m + 5 * r));
        int64_t *out = meta.mutable_data();
        for (const auto *part : {&v_src, &v_dst, &k_src, &k_dst, &k_delta}) {
            std::copy(part->begin(), part->end(), out);
            out += part->size();
        }
        for (size_t i = 0; i < r; ++i) *out++ = static_cast<int64_t>(i);
        std::fill(out, out + r, int64_t{0});
        return py::make_tuple(meta, m, r, reload, new_counts, skipped);
    }, py::arg("chosen"), py::arg("sources"), py::arg("counts"), py::arg("limit"), py::arg("untrusted"));
    // Prefill plan helpers (KVMEM_NATIVE_PREFILL_PLAN). Exact ports of the pure
    // integer Python helpers action_attention.window_segments (over a frozen
    // PrefillPlanFrame record tuple) and LogicalScorePlan's validation. Every
    // value must be an exact int within +-2^61 so int64 arithmetic equals
    // Python's; otherwise they return None/False and the caller runs Python.
    struct SmallInts {
        static bool get(PyObject *o, int64_t &out) {
            const int64_t limit = int64_t{1} << 61;
            if (!PyLong_CheckExact(o)) return false;
            int overflow = 0;
            const long long v = PyLong_AsLongLongAndOverflow(o, &overflow);
            if (overflow || v > limit || v < -limit) return false;
            out = v;
            return true;
        }
        static int64_t floor_mod(int64_t a, int64_t b) {   // Python a % b, b > 0
            const int64_t r = a % b;
            return r < 0 ? r + b : r;
        }
        static py::list to_list(const std::vector<int64_t> &values) {
            py::list out(values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                PyObject *v = PyLong_FromLongLong(values[i]);
                if (!v) throw py::error_already_set();
                PyList_SET_ITEM(out.ptr(), static_cast<Py_ssize_t>(i), v);
            }
            return out;
        }
    };
    m.def("frame_window_segments", [](py::handle records, py::handle history_begin_o, py::handle local_begin_o,
            py::handle count_o, py::handle block_o) -> py::object {
        int64_t history_begin, local_begin, count, block;
        if (!PyTuple_CheckExact(records.ptr()) || !SmallInts::get(history_begin_o.ptr(), history_begin) ||
                !SmallInts::get(local_begin_o.ptr(), local_begin) || !SmallInts::get(count_o.ptr(), count) ||
                !SmallInts::get(block_o.ptr(), block) || block <= 0)
            return py::none();
        std::vector<int64_t> starts, counts, history;
        const Py_ssize_t size = PyTuple_GET_SIZE(records.ptr());
        starts.reserve(static_cast<size_t>(size) + 64);
        counts.reserve(static_cast<size_t>(size) + 64);
        history.reserve(static_cast<size_t>(size) + 64);
        int64_t cursor = 0;
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *record = PyTuple_GET_ITEM(records.ptr(), i);
            int64_t begin, n, position;
            if (!PyTuple_CheckExact(record) || PyTuple_GET_SIZE(record) != 5 ||
                    !SmallInts::get(PyTuple_GET_ITEM(record, 2), begin) ||
                    !SmallInts::get(PyTuple_GET_ITEM(record, 3), n) ||
                    !SmallInts::get(PyTuple_GET_ITEM(record, 4), position))
                return py::none();
            if (position >= local_begin) break;
            n = std::min(n, local_begin - position);
            if (position != cursor || n <= 0 || n > block)
                throw py::value_error("Action window has a gap or stale dense mapping");
            starts.push_back(cursor);
            counts.push_back(n);
            history.push_back(begin - SmallInts::floor_mod(begin, block));
            cursor += n;
        }
        int64_t logical = history_begin + cursor - local_begin;
        if (logical < 0) throw py::value_error("Invalid action suffix origin");
        const int64_t stop = local_begin + count;
        while (cursor < stop) {
            const int64_t n = std::min(block - SmallInts::floor_mod(logical, block), stop - cursor);
            starts.push_back(cursor);
            counts.push_back(n);
            history.push_back(logical - SmallInts::floor_mod(logical, block));
            cursor += n;
            logical += n;
        }
        return py::make_tuple(SmallInts::to_list(starts), SmallInts::to_list(counts), SmallInts::to_list(history));
    }, py::arg("records"), py::arg("history_begin"), py::arg("local_begin"), py::arg("count"),
       py::arg("block_tokens"));
    // KVMEM_NATIVE_DECODE_SEGMENTS: layerwise_decode.segment_identities in one
    // call (window_segments over the layer's records, then the partition of the
    // native score columns). Returns the identity list, or None whenever any
    // input is outside the exact int64 domain or any check would raise: the
    // caller then runs the Python helper, which raises its original error.
    m.def("decode_segment_identities", [](py::handle records, py::handle history_begin_o, py::handle local_begin_o,
            py::handle count_o, py::handle starts_o, py::handle counts_o, py::handle block_o) -> py::object {
        int64_t history_begin, local_begin, count, block;
        if (!SmallInts::get(history_begin_o.ptr(), history_begin) ||
                !SmallInts::get(local_begin_o.ptr(), local_begin) || !SmallInts::get(count_o.ptr(), count) ||
                !SmallInts::get(block_o.ptr(), block) || block <= 0)
            return py::none();
        // Score segment vectors: 1-D signed/<=32-bit unsigned integer arrays, or
        // tuples/lists of exact ints. Anything else runs the Python helper.
        auto vector_of = [](py::handle o, std::vector<int64_t> &out) -> bool {
            if (py::isinstance<py::array>(o)) {
                py::array a = py::reinterpret_borrow<py::array>(o);
                const char kind = a.dtype().kind();
                if (a.ndim() != 1 || !(kind == 'i' || (kind == 'u' && a.itemsize() <= 4))) return false;
                auto c = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(a);
                if (!c) { PyErr_Clear(); return false; }
                const int64_t *p = c.data();
                out.assign(p, p + c.size());
                return true;
            }
            if (!PyList_CheckExact(o.ptr()) && !PyTuple_CheckExact(o.ptr())) return false;
            py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(o.ptr(), "segments"));
            if (!fast) { PyErr_Clear(); return false; }
            const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.ptr());
            PyObject **items = PySequence_Fast_ITEMS(fast.ptr());
            out.resize(static_cast<size_t>(n));
            for (Py_ssize_t i = 0; i < n; ++i)
                if (!SmallInts::get(items[i], out[static_cast<size_t>(i)])) return false;
            return true;
        };
        std::vector<int64_t> starts, counts;
        if (!vector_of(starts_o, starts) || !vector_of(counts_o, counts) || starts.size() != counts.size())
            return py::none();
        const int64_t limit = int64_t{1} << 61;
        for (size_t i = 0; i < starts.size(); ++i)
            if (starts[i] > limit || starts[i] < -limit || counts[i] > limit || counts[i] < -limit)
                return py::none();
        if (!PyList_CheckExact(records.ptr()) && !PyTuple_CheckExact(records.ptr())) return py::none();
        py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(records.ptr(), "records"));
        if (!fast) { PyErr_Clear(); return py::none(); }
        const Py_ssize_t size = PySequence_Fast_GET_SIZE(fast.ptr());
        PyObject **items = PySequence_Fast_ITEMS(fast.ptr());
        // window_segments(records, history_begin, local_begin, count, block)
        std::vector<int64_t> ls, lc, lh;
        ls.reserve(static_cast<size_t>(size) + 64); lc.reserve(static_cast<size_t>(size) + 64);
        lh.reserve(static_cast<size_t>(size) + 64);
        int64_t cursor = 0;
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *record = items[i];
            int64_t begin, n, position;
            if (!PyTuple_CheckExact(record) || PyTuple_GET_SIZE(record) != 5 ||
                    !SmallInts::get(PyTuple_GET_ITEM(record, 2), begin) ||
                    !SmallInts::get(PyTuple_GET_ITEM(record, 3), n) ||
                    !SmallInts::get(PyTuple_GET_ITEM(record, 4), position))
                return py::none();
            if (position >= local_begin) break;
            n = std::min(n, local_begin - position);
            if (position != cursor || n <= 0 || n > block) return py::none();
            ls.push_back(cursor); lc.push_back(n); lh.push_back(begin - SmallInts::floor_mod(begin, block));
            cursor += n;
        }
        int64_t logical = history_begin + cursor - local_begin;
        if (logical < 0) return py::none();
        const int64_t stop = local_begin + count;
        while (cursor < stop) {
            const int64_t n = std::min(block - SmallInts::floor_mod(logical, block), stop - cursor);
            ls.push_back(cursor); lc.push_back(n); lh.push_back(logical - SmallInts::floor_mod(logical, block));
            cursor += n;
            logical += n;
        }
        // Partition the native score columns; each must lie in one logical block.
        std::vector<int64_t> result;
        result.reserve(starts.size());
        cursor = 0;
        size_t index = 0;
        for (size_t k = 0; k < starts.size(); ++k) {
            const int64_t start = starts[k], seg = counts[k];
            if (start != cursor || seg <= 0) return py::none();
            const int64_t end = start + seg;
            bool have = false;
            int64_t identity = 0;
            while (cursor < end) {
                if (index >= ls.size() || !(ls[index] <= cursor && cursor < ls[index] + lc[index])) return py::none();
                if (have && identity != lh[index]) return py::none();
                identity = lh[index];
                have = true;
                const int64_t limit_end = ls[index] + lc[index];
                cursor = std::min(end, limit_end);
                if (cursor == limit_end) ++index;
            }
            result.push_back(identity);
        }
        if (cursor != local_begin + count || index != ls.size()) return py::none();
        return SmallInts::to_list(result);
    }, py::arg("records"), py::arg("history_begin"), py::arg("local_begin"), py::arg("count"),
       py::arg("starts"), py::arg("counts"), py::arg("block_tokens"));
    // KVMEM_NATIVE_DECODE_LAYER_MAP: a frozen per-layer map (tuple/list of
    // 5-tuples of exact ints) as an int64 [n, 3] array of (begin, size, position),
    // or None outside the exact domain. Parsed once per pressure frame.
    m.def("parse_layer_map", [](py::handle records) -> py::object {
        if (!PyList_CheckExact(records.ptr()) && !PyTuple_CheckExact(records.ptr())) return py::none();
        py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(records.ptr(), "records"));
        if (!fast) { PyErr_Clear(); return py::none(); }
        const Py_ssize_t size = PySequence_Fast_GET_SIZE(fast.ptr());
        PyObject **items = PySequence_Fast_ITEMS(fast.ptr());
        py::array_t<int64_t> out({static_cast<py::ssize_t>(size), static_cast<py::ssize_t>(3)});
        int64_t *p = out.mutable_data();
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *r = items[i];
            if (!PyTuple_CheckExact(r) || PyTuple_GET_SIZE(r) != 5 ||
                    !SmallInts::get(PyTuple_GET_ITEM(r, 2), p[3*i]) ||
                    !SmallInts::get(PyTuple_GET_ITEM(r, 3), p[3*i+1]) ||
                    !SmallInts::get(PyTuple_GET_ITEM(r, 4), p[3*i+2]))
                return py::none();
            int64_t ignored;
            if (!SmallInts::get(PyTuple_GET_ITEM(r, 0), ignored) || !SmallInts::get(PyTuple_GET_ITEM(r, 1), ignored))
                return py::none();
        }
        return std::move(out);
    }, py::arg("records"));
    // KVMEM_NATIVE_DECODE_LAYER_MAP: layerwise_decode.segment_identities over
    // LayerwisePrefillPressure._layer_records(frame) without building the record
    // list: extend_records(map, origin, history_begin, 32), the physical length
    // check against local_begin, window_segments and the column partition.
    // Returns None whenever any step would raise or leave the exact domain; the
    // caller then runs the original Python path, which raises its own error.
    m.def("decode_identities_from_map", [](py::array_t<int64_t, py::array::c_style> parsed, py::handle origin_o,
            py::handle history_begin_o, py::handle local_begin_o, py::handle count_o, py::handle starts_o,
            py::handle counts_o) -> py::object {
        const int64_t block = 32, limit = int64_t{1} << 61;
        int64_t origin, history_begin, local_begin, count;
        if (!SmallInts::get(origin_o.ptr(), origin) || !SmallInts::get(history_begin_o.ptr(), history_begin) ||
                !SmallInts::get(local_begin_o.ptr(), local_begin) || !SmallInts::get(count_o.ptr(), count))
            return py::none();
        if (parsed.ndim() != 2 || parsed.shape(1) != 3) return py::none();
        auto vector_of = [](py::handle o, std::vector<int64_t> &out) -> bool {
            if (py::isinstance<py::array>(o)) {
                py::array a = py::reinterpret_borrow<py::array>(o);
                const char kind = a.dtype().kind();
                if (a.ndim() != 1 || !(kind == 'i' || (kind == 'u' && a.itemsize() <= 4))) return false;
                auto c = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(a);
                if (!c) { PyErr_Clear(); return false; }
                out.assign(c.data(), c.data() + c.size());
                return true;
            }
            if (!PyList_CheckExact(o.ptr()) && !PyTuple_CheckExact(o.ptr())) return false;
            py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(o.ptr(), "segments"));
            if (!fast) { PyErr_Clear(); return false; }
            const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.ptr());
            PyObject **items = PySequence_Fast_ITEMS(fast.ptr());
            out.resize(static_cast<size_t>(n));
            for (Py_ssize_t i = 0; i < n; ++i)
                if (!SmallInts::get(items[i], out[static_cast<size_t>(i)])) return false;
            return true;
        };
        std::vector<int64_t> starts, counts;
        if (!vector_of(starts_o, starts) || !vector_of(counts_o, counts) || starts.size() != counts.size())
            return py::none();
        for (size_t i = 0; i < starts.size(); ++i)
            if (starts[i] > limit || starts[i] < -limit || counts[i] > limit || counts[i] < -limit) return py::none();
        // extend_records(records, origin, end=history_begin, block_tokens=32)
        if (history_begin < origin) return py::none();
        const int64_t n0 = parsed.shape(0);
        const int64_t *src = parsed.data();
        std::vector<int64_t> rb, rs, rp;
        rb.reserve(static_cast<size_t>(n0) + 160); rs.reserve(static_cast<size_t>(n0) + 160);
        rp.reserve(static_cast<size_t>(n0) + 160);
        int64_t cursor = 0;
        for (int64_t i = 0; i < n0; ++i) {
            const int64_t b = src[3*i], s = src[3*i+1], p = src[3*i+2];
            if (b > limit || b < -limit || s > limit || s < -limit || p > limit || p < -limit) return py::none();
            rb.push_back(b); rs.push_back(s); rp.push_back(p);
            cursor += s;
            if (cursor > limit || cursor < -limit) return py::none();
        }
        auto floor_div = [](int64_t a, int64_t d) { int64_t q = a / d; return (a % d != 0 && ((a < 0) != (d < 0))) ? q - 1 : q; };
        int64_t logical = origin;
        while (logical < history_begin) {
            const int64_t n = std::min(block - SmallInts::floor_mod(logical, block), history_begin - logical);
            if (!rb.empty() && rb.back() + rs.back() == logical && rp.back() + rs.back() == cursor &&
                    floor_div(rb.back(), block) == floor_div(logical, block)) {
                rs.back() += n;
            } else {
                rb.push_back(logical); rs.push_back(n); rp.push_back(cursor);
            }
            cursor += n;
            logical += n;
        }
        // LayerwisePrefillPressure._layer_records length check.
        int64_t total = 0;
        for (auto s : rs) total += s;
        if (total != local_begin) return py::none();
        // window_segments(records, history_begin, local_begin, count, 32)
        std::vector<int64_t> ls, lc, lh;
        ls.reserve(rb.size() + 64); lc.reserve(rb.size() + 64); lh.reserve(rb.size() + 64);
        cursor = 0;
        for (size_t i = 0; i < rb.size(); ++i) {
            const int64_t position = rp[i];
            if (position >= local_begin) break;
            const int64_t n = std::min(rs[i], local_begin - position);
            if (position != cursor || n <= 0 || n > block) return py::none();
            ls.push_back(cursor); lc.push_back(n); lh.push_back(rb[i] - SmallInts::floor_mod(rb[i], block));
            cursor += n;
        }
        logical = history_begin + cursor - local_begin;
        if (logical < 0) return py::none();
        const int64_t stop = local_begin + count;
        while (cursor < stop) {
            const int64_t n = std::min(block - SmallInts::floor_mod(logical, block), stop - cursor);
            ls.push_back(cursor); lc.push_back(n); lh.push_back(logical - SmallInts::floor_mod(logical, block));
            cursor += n;
            logical += n;
        }
        // Partition of the native score columns (segment_identities).
        std::vector<int64_t> result;
        result.reserve(starts.size());
        cursor = 0;
        size_t index = 0;
        for (size_t k = 0; k < starts.size(); ++k) {
            const int64_t start = starts[k], seg = counts[k];
            if (start != cursor || seg <= 0) return py::none();
            const int64_t end = start + seg;
            bool have = false;
            int64_t identity = 0;
            while (cursor < end) {
                if (index >= ls.size() || !(ls[index] <= cursor && cursor < ls[index] + lc[index])) return py::none();
                if (have && identity != lh[index]) return py::none();
                identity = lh[index];
                have = true;
                const int64_t limit_end = ls[index] + lc[index];
                cursor = std::min(end, limit_end);
                if (cursor == limit_end) ++index;
            }
            result.push_back(identity);
        }
        if (cursor != local_begin + count || index != ls.size()) return py::none();
        return SmallInts::to_list(result);
    }, py::arg("parsed"), py::arg("origin"), py::arg("history_begin"), py::arg("local_begin"), py::arg("count"),
       py::arg("starts"), py::arg("counts"));
    // KVMEM_NATIVE_DECODE_LAYER_BATCH: decode_identities_from_map for every
    // layer of one request in a single call. The score columns are read once,
    // the work vectors are reused across layers, and the identities are written
    // into one int64 [layers, columns] array instead of one Python int list per
    // layer. Returns (layer 0 as a list of ints, the read-only array), or None
    // whenever decode_identities_from_map would return None for any layer or a
    // map is not an aligned C-contiguous int64 [n, 3] array: the caller then
    // runs the per-layer path.
    m.def("decode_identities_from_maps", [](py::handle maps_o, py::handle origin_o, py::handle history_begin_o,
            py::handle local_begin_o, py::handle count_o, py::handle starts_o, py::handle counts_o) -> py::object {
        const int64_t block = 32, limit = int64_t{1} << 61;
        int64_t origin, history_begin, local_begin, count;
        if (!SmallInts::get(origin_o.ptr(), origin) || !SmallInts::get(history_begin_o.ptr(), history_begin) ||
                !SmallInts::get(local_begin_o.ptr(), local_begin) || !SmallInts::get(count_o.ptr(), count))
            return py::none();
        if (!PyList_CheckExact(maps_o.ptr()) && !PyTuple_CheckExact(maps_o.ptr())) return py::none();
        py::object maps = py::reinterpret_steal<py::object>(PySequence_Fast(maps_o.ptr(), "maps"));
        if (!maps) { PyErr_Clear(); return py::none(); }
        const Py_ssize_t layers = PySequence_Fast_GET_SIZE(maps.ptr());
        PyObject **map_items = PySequence_Fast_ITEMS(maps.ptr());
        if (layers <= 0) return py::none();
        auto vector_of = [](py::handle o, std::vector<int64_t> &out) -> bool {
            if (py::isinstance<py::array>(o)) {
                py::array a = py::reinterpret_borrow<py::array>(o);
                const char kind = a.dtype().kind();
                if (a.ndim() != 1 || !(kind == 'i' || (kind == 'u' && a.itemsize() <= 4))) return false;
                auto c = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(a);
                if (!c) { PyErr_Clear(); return false; }
                out.assign(c.data(), c.data() + c.size());
                return true;
            }
            if (!PyList_CheckExact(o.ptr()) && !PyTuple_CheckExact(o.ptr())) return false;
            py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(o.ptr(), "segments"));
            if (!fast) { PyErr_Clear(); return false; }
            const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.ptr());
            PyObject **items = PySequence_Fast_ITEMS(fast.ptr());
            out.resize(static_cast<size_t>(n));
            for (Py_ssize_t i = 0; i < n; ++i)
                if (!SmallInts::get(items[i], out[static_cast<size_t>(i)])) return false;
            return true;
        };
        std::vector<int64_t> starts, counts;
        if (!vector_of(starts_o, starts) || !vector_of(counts_o, counts) || starts.size() != counts.size())
            return py::none();
        for (size_t i = 0; i < starts.size(); ++i)
            if (starts[i] > limit || starts[i] < -limit || counts[i] > limit || counts[i] < -limit) return py::none();
        if (history_begin < origin) return py::none();
        const size_t columns = starts.size();
        py::array_t<int64_t> out({static_cast<py::ssize_t>(layers), static_cast<py::ssize_t>(columns)});
        int64_t *rows = out.mutable_data();
        auto floor_div = [](int64_t a, int64_t d) { int64_t q = a / d; return (a % d != 0 && ((a < 0) != (d < 0))) ? q - 1 : q; };
        // The three steps of decode_identities_from_map run as one pass per
        // layer: the frame's rows are read in place (only its last row and the
        // generated extension are materialized), window segments are produced
        // one at a time, and each is consumed by the score-column partition.
        // Every check of the per-layer function is still made on the same
        // values; a failed check anywhere returns None.
        const int64_t stop = local_begin + count;
        // Python objects are read only here, with the GIL: each layer's map rows
        // (kept alive by maps). A map of the wrong type or layout returns None.
        std::vector<const int64_t *> sources(static_cast<size_t>(layers));
        std::vector<int64_t> lengths(static_cast<size_t>(layers));
        for (Py_ssize_t layer = 0; layer < layers; ++layer) {
            if (!py::isinstance<py::array_t<int64_t>>(map_items[layer])) return py::none();
            py::array parsed = py::reinterpret_borrow<py::array>(map_items[layer]);
            const int need = py::array::c_style | py::detail::npy_api::NPY_ARRAY_ALIGNED_;
            if (parsed.ndim() != 2 || parsed.shape(1) != 3 || (parsed.flags() & need) != need) return py::none();
            sources[static_cast<size_t>(layer)] = static_cast<const int64_t *>(parsed.data());
            lengths[static_cast<size_t>(layer)] = parsed.shape(0);
        }
        // One layer's pass, false wherever the per-layer function returns None.
        // Layers are independent and each writes only its own row of out, so
        // they run on the publication pool (KVMEM_NATIVE_PUBLISH_THREADS)
        // without the GIL; the array is the same for any thread count.
        auto decode_layer = [&](size_t layer) -> bool {
            thread_local std::vector<int64_t> eb, es, ep;
            // extend_records(records, origin, end=history_begin, block_tokens=32)
            const int64_t n0 = lengths[layer];
            const int64_t *src = sources[layer];
            int64_t cursor = 0;
            for (int64_t i = 0; i < n0; ++i) {
                const int64_t b = src[3*i], s = src[3*i+1], p = src[3*i+2];
                if (b > limit || b < -limit || s > limit || s < -limit || p > limit || p < -limit) return false;
                cursor += s;
                if (cursor > limit || cursor < -limit) return false;
            }
            // Rows [0, kept) stay in the frame; its last row (the only one an
            // extension can lengthen) and the extension rows live in eb/es/ep.
            const int64_t kept = n0 > 0 ? n0 - 1 : 0;
            eb.clear(); es.clear(); ep.clear();
            if (n0 > 0) { eb.push_back(src[3*kept]); es.push_back(src[3*kept+1]); ep.push_back(src[3*kept+2]); }
            int64_t logical = origin;
            while (logical < history_begin) {
                const int64_t n = std::min(block - SmallInts::floor_mod(logical, block), history_begin - logical);
                if (!eb.empty() && eb.back() + es.back() == logical && ep.back() + es.back() == cursor &&
                        floor_div(eb.back(), block) == floor_div(logical, block)) {
                    es.back() += n;
                } else {
                    eb.push_back(logical); es.push_back(n); ep.push_back(cursor);
                }
                cursor += n;
                logical += n;
            }
            // LayerwisePrefillPressure._layer_records length check: the sum of
            // every row's size, which is what cursor has accumulated.
            if (cursor != local_begin) return false;
            // window_segments(records, history_begin, local_begin, count, 32),
            // one segment per call: 1 = produced, 0 = no more, -1 = invalid.
            const int64_t records = kept + static_cast<int64_t>(eb.size());
            int64_t record = 0, made = 0, tail_logical = 0;
            bool tail = false;
            int64_t seg_start = 0, seg_size = 0, seg_block = 0;
            auto next = [&]() -> int {
                if (!tail) {
                    if (record < records) {
                        int64_t b, size, position;
                        if (record < kept) { b = src[3*record]; size = src[3*record+1]; position = src[3*record+2]; }
                        else { const size_t e = static_cast<size_t>(record - kept); b = eb[e]; size = es[e]; position = ep[e]; }
                        if (position < local_begin) {
                            const int64_t n = std::min(size, local_begin - position);
                            if (position != made || n <= 0 || n > block) return -1;
                            seg_start = made; seg_size = n; seg_block = b - SmallInts::floor_mod(b, block);
                            made += n;
                            ++record;
                            return 1;
                        }
                    }
                    tail = true;
                    tail_logical = history_begin + made - local_begin;
                    if (tail_logical < 0) return -1;
                }
                if (made < stop) {
                    const int64_t n = std::min(block - SmallInts::floor_mod(tail_logical, block), stop - made);
                    seg_start = made; seg_size = n; seg_block = tail_logical - SmallInts::floor_mod(tail_logical, block);
                    made += n;
                    tail_logical += n;
                    return 1;
                }
                return 0;
            };
            // Partition of the native score columns (segment_identities).
            int64_t *result = rows + layer * columns;
            int state = next();
            if (state < 0) return false;
            cursor = 0;
            for (size_t k = 0; k < columns; ++k) {
                const int64_t start = starts[k], seg = counts[k];
                if (start != cursor || seg <= 0) return false;
                const int64_t end = start + seg;
                bool have = false;
                int64_t identity = 0;
                while (cursor < end) {
                    if (state == 0 || !(seg_start <= cursor && cursor < seg_start + seg_size)) return false;
                    if (have && identity != seg_block) return false;
                    identity = seg_block;
                    have = true;
                    const int64_t limit_end = seg_start + seg_size;
                    cursor = std::min(end, limit_end);
                    if (cursor == limit_end) {
                        state = next();
                        if (state < 0) return false;
                    }
                }
                result[k] = identity;
            }
            // Every segment was consumed (and so produced and checked).
            return cursor == stop && state == 0;
        };
        std::vector<char> ok(static_cast<size_t>(layers), 0);
        {
            py::gil_scoped_release release;
            kvmem_parallel::for_rows(static_cast<size_t>(layers), [&](size_t layer) { ok[layer] = decode_layer(layer); });
        }
        for (char done : ok)
            if (!done) return py::none();
        py::list first(columns);
        for (size_t k = 0; k < columns; ++k) {
            PyObject *v = PyLong_FromLongLong(rows[k]);
            if (!v) throw py::error_already_set();
            PyList_SET_ITEM(first.ptr(), static_cast<Py_ssize_t>(k), v);
        }
        py::detail::array_proxy(out.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return py::make_tuple(first, out);
    }, py::arg("maps"), py::arg("origin"), py::arg("history_begin"), py::arg("local_begin"), py::arg("count"),
       py::arg("starts"), py::arg("counts"));
    // LogicalScorePlan.__post_init__ after its span isinstance check. Returns
    // None (valid), the ValueError message to raise, or False (run Python).
    m.def("logical_plan_check", [](py::handle rows, py::handle starts, py::handle counts, py::handle history,
            py::handle block_o, py::handle span_count_o, py::handle local_end_o) -> py::object {
        const char *names[] = {"rows", "starts", "counts", "history"};
        PyObject *vectors[] = {rows.ptr(), starts.ptr(), counts.ptr(), history.ptr()};
        for (int k = 0; k < 4; ++k) {
            bool ok = PyTuple_CheckExact(vectors[k]);
            for (Py_ssize_t i = 0; ok && i < PyTuple_GET_SIZE(vectors[k]); ++i)
                ok = PyLong_CheckExact(PyTuple_GET_ITEM(vectors[k], i));
            if (!ok) return py::str(std::string(names[k]) + " must be an immutable tuple of integers");
        }
        int64_t block, span_count, local_end;
        if (!PyLong_CheckExact(block_o.ptr()))
            return py::str("Logical score block size must be positive");
        if (!SmallInts::get(block_o.ptr(), block) || !SmallInts::get(span_count_o.ptr(), span_count) ||
                !SmallInts::get(local_end_o.ptr(), local_end))
            return py::bool_(false);
        if (block <= 0) return py::str("Logical score block size must be positive");
        std::vector<int64_t> values[4];
        for (int k = 0; k < 4; ++k) {
            const Py_ssize_t size = PyTuple_GET_SIZE(vectors[k]);
            values[k].resize(static_cast<size_t>(size));
            for (Py_ssize_t i = 0; i < size; ++i)
                if (!SmallInts::get(PyTuple_GET_ITEM(vectors[k], i), values[k][static_cast<size_t>(i)]))
                    return py::bool_(false);
        }
        const auto &r = values[0], &s = values[1], &c = values[2], &h = values[3];
        bool bad_rows = false;
        for (size_t i = 0; i < r.size() && !bad_rows; ++i)
            bad_rows = !(0 <= r[i] && r[i] < span_count) || (i + 1 < r.size() && r[i] >= r[i + 1]);
        if (bad_rows) return py::str("Logical sampled rows must be increasing and inside their original span");
        if (s.empty() || s.size() != c.size() || c.size() != h.size())
            return py::str("Logical segment vectors must have equal nonzero lengths");
        int64_t cursor = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] != cursor || !(0 < c[i] && c[i] <= block) || h[i] < 0 || SmallInts::floor_mod(h[i], block))
                return py::str("Logical segments must partition the visible context with history page identities");
            cursor += c[i];
        }
        if (cursor != local_end)
            return py::str("Logical segments must cover exactly their original visible context");
        return py::none();
    }, py::arg("rows"), py::arg("starts"), py::arg("counts"), py::arg("history"), py::arg("block_tokens"),
       py::arg("span_count"), py::arg("local_end"));
    // Identity table for pressure commit maps (KVMEM_PRESSURE_NATIVE_COMMIT_TABLE):
    // (slot, generation, history_begin, count) per accepted record, kept in the
    // record index's sorted order and extended as records are appended. build()
    // reproduces pressure_commit.prepare_commit's maps: the lookups, checks and
    // offsets run without the GIL; the row tuples are created one layer at a time
    // with the GIL briefly released between layers, so the engine thread is
    // never held for a whole commit.
    struct IdentityTable {
        std::vector<uint32_t> slot;
        std::vector<py::object> generation, begin, count;   // the records' own int objects
        std::vector<int64_t> begin_i, count_i;
        std::unordered_map<uint32_t, uint32_t> position;
        py::object last;                                     // last indexed Record (prefix check)
        std::mutex mu;                                       // extend vs. GIL-free lookups
    };
    py::class_<IdentityTable>(m, "IdentityTable")
        .def(py::init<>())
        .def("__len__", [](const IdentityTable &t) { return t.slot.size(); })
        .def("matches", [](const IdentityTable &t, const py::list &records) {
            // The table indexes a prefix of records (same objects, same order).
            const size_t n = t.slot.size();
            if (n > records.size()) return false;
            if (n == 0) return true;
            return records[n - 1].ptr() == t.last.ptr();
        })
        .def("extend", [](IdentityTable &t, const py::list &records, const py::list &slots) {
            if (records.size() != slots.size())
                throw py::value_error("Identity table needs one slot per record");
            py::str generation_name("generation"), handle_name("handle"), begin_name("history_begin"),
                count_name("count");
            for (size_t i = t.slot.size(); i < records.size(); ++i) {
                py::handle record = records[i];
                py::object handle = py::reinterpret_steal<py::object>(PyObject_GetAttr(record.ptr(), handle_name.ptr()));
                if (!handle) throw py::error_already_set();
                py::object generation = py::reinterpret_steal<py::object>(PyObject_GetAttr(handle.ptr(), generation_name.ptr()));
                if (!generation) throw py::error_already_set();
                py::object begin = py::reinterpret_steal<py::object>(PyObject_GetAttr(record.ptr(), begin_name.ptr()));
                if (!begin) throw py::error_already_set();
                py::object count = py::reinterpret_steal<py::object>(PyObject_GetAttr(record.ptr(), count_name.ptr()));
                if (!count) throw py::error_already_set();
                const auto slot = slots[i].cast<uint32_t>();
                std::lock_guard<std::mutex> lock(t.mu);
                if (!t.position.emplace(slot, static_cast<uint32_t>(t.slot.size())).second)
                    throw std::runtime_error("Pressure archive contains duplicate physical slots");
                t.slot.push_back(slot);
                t.generation.push_back(generation);
                t.begin_i.push_back(begin.cast<int64_t>());
                t.count_i.push_back(count.cast<int64_t>());
                t.begin.push_back(begin);
                t.count.push_back(count);
                t.last = py::reinterpret_borrow<py::object>(record);
            }
        })
        .def("build", [](const IdentityTable &t, const std::vector<std::vector<uint32_t>> &selected,
                         const py::list &chosen, const py::list &records) {
            // prepare_commit: identities of every selected slot must exist.
            const size_t layers = selected.size();
            if (layers != chosen.size() || layers == 0)
                throw std::runtime_error("Prepared pressure maps must cover all selected layers");
            std::vector<std::vector<uint32_t>> positions(layers);
            std::vector<int64_t> lengths(layers, 0);
            bool missing = false;
            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(const_cast<IdentityTable &>(t).mu);
                for (size_t l = 0; l < layers; ++l) {
                    positions[l].reserve(selected[l].size());
                    for (const auto s : selected[l]) {
                        const auto it = t.position.find(s);
                        if (it == t.position.end()) { missing = true; break; }
                        positions[l].push_back(it->second);
                        lengths[l] += t.count_i[it->second];
                    }
                    if (missing) break;
                }
            }
            if (missing)
                throw std::runtime_error("Selected pressure identity is missing from the accepted archive");
            // Each layer's picked records are the indexed records at those positions.
            for (size_t l = 0; l < layers; ++l) {
                py::list picked = py::reinterpret_borrow<py::list>(chosen[l]);
                if (picked.size() != positions[l].size())
                    throw std::runtime_error("Prepared pressure map differs from the selected record order");
                for (size_t i = 0; i < positions[l].size(); ++i) {
                    if (picked[i].ptr() != records[positions[l][i]].ptr()) {
                        const int differs = PyObject_RichCompareBool(
                            py::tuple(py::make_tuple(picked[i].attr("history_begin"), picked[i].attr("count"))).ptr(),
                            py::make_tuple(t.begin[positions[l][i]], t.count[positions[l][i]]).ptr(), Py_NE);
                        if (differs < 0) throw py::error_already_set();
                        if (differs)
                            throw std::runtime_error("Prepared pressure map differs from the selected record order");
                    }
                }
            }
            py::list maps;
            for (size_t l = 0; l < layers; ++l) {
                const auto &pos = positions[l];
                PyObject *rows = PyTuple_New(static_cast<Py_ssize_t>(pos.size()));
                if (!rows) throw py::error_already_set();
                py::object owner = py::reinterpret_steal<py::object>(rows);
                int64_t offset = 0;
                for (size_t i = 0; i < pos.size(); ++i) {
                    const uint32_t p = pos[i];
                    PyObject *slot = PyLong_FromUnsignedLong(t.slot[p]);
                    PyObject *off = PyLong_FromLongLong(offset);
                    if (!slot || !off) { Py_XDECREF(slot); Py_XDECREF(off); throw py::error_already_set(); }
                    PyObject *row = PyTuple_Pack(5, slot, t.generation[p].ptr(), t.begin[p].ptr(), t.count[p].ptr(), off);
                    Py_DECREF(slot); Py_DECREF(off);
                    if (!row) throw py::error_already_set();
                    PyTuple_SET_ITEM(rows, static_cast<Py_ssize_t>(i), row);
                    offset += t.count_i[p];
                }
                maps.append(owner);
                {   // let a waiting engine thread take the GIL between layers
                    py::gil_scoped_release release;
                    std::this_thread::yield();
                }
            }
            for (size_t l = 1; l < layers; ++l)
                if (lengths[l] != lengths[0])
                    throw std::runtime_error("Prepared pressure maps changed the per-layer token extent");
            return py::make_tuple(py::tuple(maps), lengths[0]);
        }, py::arg("selected"), py::arg("chosen"), py::arg("records"));
    // all(id(r) in table for r in records)
    m.def("prepared_contains", [](const py::iterable &records, const py::dict &table) {
        for (const auto &record : records) {
            py::object key = py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(record.ptr()));
            if (!key) throw py::error_already_set();
            const int found = PyDict_Contains(table.ptr(), key.ptr());
            if (found < 0) throw py::error_already_set();
            if (!found) return false;
        }
        return true;
    }, py::arg("records"), py::arg("table"));
    py::class_<StreamPages>(m, "StreamPages")
        .def(py::init<py::buffer, py::buffer, const std::vector<py::buffer> &, const std::vector<py::buffer> &,
                      const std::vector<py::buffer> &, ssize_t>(),
             py::arg("source_raw"), py::arg("source_values"), py::arg("page_raw"), py::arg("page_values"),
             py::arg("page_sums"), py::arg("block"))
        .def("layer", &StreamPages::layer, py::arg("layer"), py::arg("vectorized") = false)
        .def_property_readonly("layers", &StreamPages::layers)
        .def_property_readonly("scoring_layers", &StreamPages::scoring);
    m.def("finalize_mean_batch", [](const std::vector<py::tuple> &jobs, bool vectorized) -> py::object {
        // Validate every page before touching any sum. Buffer exports retain
        // their owners across one GIL release for the accepted packet, instead
        // of reacquiring the engine GIL after each small sealed page.
        struct Row { py::buffer_info raw, sums; ssize_t count; bool fp8; };
        std::vector<Row> rows;
        rows.reserve(jobs.size());
        for (const auto &job : jobs) {
            if (job.size()!=4) throw std::invalid_argument("Invalid content-index job");
            auto a=py::cast<py::buffer>(job[0]).request();
            auto b=py::cast<py::buffer>(job[1]).request(true);
            const auto count=py::cast<ssize_t>(job[2]);
            const auto fp8=py::cast<bool>(job[3]);
            const ssize_t width=fp8 ? 1 : 2;
            if (a.ndim!=4 || a.itemsize!=1 || b.ndim!=3 || b.itemsize!=4 || b.readonly ||
                b.format!=py::format_descriptor<float>::format() || count<=0 || count>a.shape[1] ||
                b.shape[0]<=0 || b.shape[0]>a.shape[0] || b.shape[1]!=a.shape[2] ||
                b.shape[2]*width!=a.shape[3] || b.shape[1]<=0 || b.shape[2]<=0 ||
                a.strides[3]!=1 || a.strides[2]!=a.shape[3] ||
                a.strides[1]<a.shape[2]*a.shape[3] || a.strides[0]<a.shape[1]*a.strides[1] ||
                b.strides[2]!=4 || b.strides[1]!=b.shape[2]*4 || b.strides[0]!=b.shape[1]*b.strides[1])
                throw std::invalid_argument("Invalid content-index buffers");
            rows.push_back({std::move(a),std::move(b),count,fp8});
        }
        std::vector<uint8_t> vectorized_rows(rows.size(), 0);
        // Rows may run on several threads only when no two sums overlap; then
        // every page keeps its own serial token-order FP32 additions.
        std::vector<const py::buffer_info *> written, read;
        written.reserve(rows.size());
        read.reserve(rows.size());
        for (const auto &r : rows) {
            written.push_back(&r.sums);
            read.push_back(&r.raw);
        }
        const bool disjoint = disjoint_buffers(written, read);
        {
          py::gil_scoped_release release;
          auto finalize_row = [&](size_t index) {
            const auto &r = rows[index];
            const auto &a=r.raw, &b=r.sums;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
            if (vectorized && r.fp8 && mean_fp8_simd_eligible(a, b, r.count)) {
                finalize_mean_fp8_simd(a, b, r.count);
                vectorized_rows[index] = 1;
                return;
            }
#endif
            for (ssize_t layer=0;layer<b.shape[0];++layer) {
                auto *out=reinterpret_cast<float*>(static_cast<char*>(b.ptr)+layer*b.strides[0]);
                const ssize_t channels=b.shape[1]*b.shape[2];
                std::fill(out,out+channels,0.f);
                for (ssize_t token=0;token<r.count;++token) {
                    const char *row=static_cast<const char*>(a.ptr)+layer*a.strides[0]+token*a.strides[1];
                    if (r.fp8) {
                        for (ssize_t channel=0;channel<channels;++channel)
                            out[channel]+=fp8_as_float(row+channel);
                    } else {
                        for (ssize_t channel=0;channel<channels;++channel)
                            out[channel]+=bf16_as_float(row+channel*2);
                    }
                }
            }
          };
          if (disjoint) {
            kvmem_parallel::for_rows(rows.size(), finalize_row);
          } else {
            for (size_t index = 0; index < rows.size(); ++index) finalize_row(index);
          }
        }
        size_t vectorized_records = 0;
        for (const auto flag : vectorized_rows) vectorized_records += flag;
        if (vectorized) return py::int_(vectorized_records);
        return py::none();
    }, py::arg("jobs"), py::arg("vectorized") = false);
    m.def("scatter_history_batch", [](const std::vector<py::tuple> &jobs) -> py::object {
        struct Row {
            py::buffer_info raw, values, dst_raw, dst_values, delta, sum;
            ssize_t begin, offset, count;
            bool copy_only, finalize_mean;
        };
        std::vector<Row> rows;
        rows.reserve(jobs.size());
        for (const auto &job : jobs) {
            if (job.size() != 9 && job.size() != 10 && job.size() != 11)
                throw std::invalid_argument("Invalid history scatter descriptor");
            if (job.size() >= 10 && !PyBool_Check(job[9].ptr()))
                throw std::invalid_argument("History copy-only flag must be a bool");
            if (job.size() == 11 && !PyBool_Check(job[10].ptr()))
                throw std::invalid_argument("History finalize-mean flag must be a bool");
            const bool copy_only = job.size() >= 10 && job[9].cast<bool>();
            const bool finalize_mean = job.size() == 11 && job[10].cast<bool>();
            if (finalize_mean && !copy_only)
                throw std::invalid_argument("Fused mean requires a completed copy-only page");
            if (copy_only && !job[4].is_none())
                throw std::invalid_argument("History copy-only descriptor cannot contain a reduction");
            Row r{job[0].cast<py::buffer>().request(), job[1].cast<py::buffer>().request(),
                  job[2].cast<py::buffer>().request(), job[3].cast<py::buffer>().request(),
                  job[4].is_none() ? py::buffer_info{} : job[4].cast<py::buffer>().request(),
                  job[5].cast<py::buffer>().request(),
                  job[6].cast<ssize_t>(), job[7].cast<ssize_t>(), job[8].cast<ssize_t>(),
                  copy_only, finalize_mean};
            auto byte_rows = [](const py::buffer_info &v, bool writable) {
                if (v.ndim != 4 || v.itemsize != 1 || (writable && v.readonly)) return false;
                if (v.shape[0] <= 0 || v.shape[1] <= 0 || v.shape[2] <= 0 || v.shape[3] <= 0) return false;
                return v.strides[3] == 1 && v.strides[2] == v.shape[3] &&
                       v.strides[1] >= v.shape[2] * v.shape[3] &&
                       v.strides[0] >= v.shape[2] * v.shape[3];
            };
            auto sums = [](const py::buffer_info &v, bool writable) {
                return v.ndim == 3 && v.format == py::format_descriptor<float>::format() &&
                       v.itemsize == 4 && (!writable || !v.readonly) &&
                       v.strides[2] == 4 && v.strides[1] == v.shape[2] * 4 &&
                       v.strides[0] == v.shape[1] * v.strides[1];
            };
            if (!byte_rows(r.raw, false) || !byte_rows(r.values, false) ||
                !byte_rows(r.dst_raw, true) || !byte_rows(r.dst_values, true) ||
                !sums(r.sum, true) ||
                (r.delta.ptr && (!sums(r.delta, false) || r.delta.shape != r.sum.shape)) ||
                (!r.copy_only && !r.delta.ptr && (!native_decode_sum_supported() || r.count > 4 ||
                                 (r.sum.shape[1] * r.sum.shape[2]) % 64)) ||
                r.sum.shape[0] <= 0 ||
                r.sum.shape[0] > r.raw.shape[0] || r.sum.shape[1] != r.values.shape[2] ||
                r.sum.shape[2] != r.values.shape[3] || r.begin < 0 || r.offset < 0 || r.count <= 0)
                throw std::invalid_argument("Invalid history scatter geometry or buffer ownership");
            const ssize_t raw_bytes=r.raw.shape[3]/r.values.shape[3];
            if ((raw_bytes!=1 && raw_bytes!=2) || (!r.delta.ptr && raw_bytes==2 &&
                    r.raw.shape[3]!=r.values.shape[3]*2))
                throw std::invalid_argument("Unsupported history raw-K element width");
            for (ssize_t axis : {0, 2, 3}) {
                if (r.values.shape[axis] != r.dst_values.shape[axis] ||
                    r.raw.shape[axis] != r.dst_raw.shape[axis] ||
                    r.raw.shape[axis] != r.values.shape[axis] * (axis == 3 ? raw_bytes : 1))
                    throw std::invalid_argument("History scatter codec mismatch");
            }
            if (r.raw.shape[1] != r.values.shape[1] ||
                r.dst_raw.shape[1] != r.dst_values.shape[1] ||
                r.count > r.raw.shape[1] || r.begin > r.raw.shape[1] - r.count ||
                r.count > r.dst_raw.shape[1] || r.offset > r.dst_raw.shape[1] - r.count)
                throw std::invalid_argument("History scatter extent outside owned rows");
            if (r.copy_only &&
                (r.offset != r.dst_raw.shape[1] - r.count ||
                 r.dst_raw.strides[0] < r.dst_raw.shape[1] * r.dst_raw.strides[1] ||
                 r.raw.format != py::format_descriptor<uint8_t>::format() ||
                 r.values.format != py::format_descriptor<uint8_t>::format() ||
                 r.dst_raw.format != py::format_descriptor<uint8_t>::format() ||
                 r.dst_values.format != py::format_descriptor<uint8_t>::format()))
                throw std::invalid_argument("History copy-only requires a completed byte-codec page");
            rows.push_back(std::move(r));
        }
        // Moving the final sum before later copies is safe only for disjoint
        // destinations. Reject aliases before *any* write; retain the original
        // acceptance/ordering for callers which do not request this fusion.
        auto overlaps = [](const py::buffer_info &a, const py::buffer_info &b) {
            if (!a.ptr || !b.ptr) return false;
            auto extent = [](const py::buffer_info &v) {
                size_t bytes = v.itemsize;
                for (ssize_t axis = 0; axis < v.ndim; ++axis)
                    bytes += (v.shape[axis] - 1) * v.strides[axis];
                return bytes;
            };
            const auto x = reinterpret_cast<uintptr_t>(a.ptr);
            const auto y = reinterpret_cast<uintptr_t>(b.ptr);
            return x <= y ? y - x < extent(a) : x - y < extent(b);
        };
        for (const auto &r : rows) if (r.finalize_mean) {
            for (const auto &other : rows) {
                for (const auto *buffer : {&other.raw, &other.values, &other.dst_raw,
                                          &other.dst_values, &other.delta})
                    if (overlaps(r.sum, *buffer))
                        throw std::invalid_argument("Fused mean sum aliases history buffers");
                if (&other != &r && (overlaps(r.sum, other.sum) ||
                        overlaps(r.dst_raw, other.dst_raw) ||
                        overlaps(r.dst_raw, other.dst_values) ||
                        overlaps(r.dst_raw, other.sum)))
                    throw std::invalid_argument("Fused mean page aliases another destination");
            }
            if (overlaps(r.dst_raw, r.dst_values))
                throw std::invalid_argument("Fused mean raw and value pages alias");
        }
        // Every descriptor is validated before touching any destination. The
        // exported buffers retain the packet and each authority through this call.
        size_t finalized_records = 0;
        // Copy-only prefill pages with disjoint destinations are independent;
        // opt-in KVMEM_NATIVE_PUBLISH_THREADS spreads them across threads.
        // Reductions into shared or aliased sums keep the serial row order.
        bool parallel = true;
        std::vector<const py::buffer_info *> written, read;
        for (const auto &r : rows) {
            parallel = parallel && r.copy_only;
            written.push_back(&r.dst_raw);
            written.push_back(&r.dst_values);
            if (r.finalize_mean) written.push_back(&r.sum);
            read.push_back(&r.raw);
            read.push_back(&r.values);
        }
        parallel = parallel && rows.size() > 1 && disjoint_buffers(written, read);
        {
        py::gil_scoped_release no_gil;
        auto scatter_row = [&](size_t index) {
            const auto &r = rows[index];
            for (ssize_t layer = 0; layer < r.raw.shape[0]; ++layer) {
                auto copy_rows = [&](const py::buffer_info &dst, const py::buffer_info &src) {
                    const auto width = src.shape[2] * src.shape[3];
                    auto *out = static_cast<char *>(dst.ptr) + layer * dst.strides[0] + r.offset * dst.strides[1];
                    const auto *in = static_cast<const char *>(src.ptr) + layer * src.strides[0] + r.begin * src.strides[1];
                    if (src.strides[1] == width && dst.strides[1] == width) {
                        std::memcpy(out, in, r.count * width);
                    } else {
                        for (ssize_t token = 0; token < r.count; ++token)
                            std::memcpy(out + token * dst.strides[1], in + token * src.strides[1], width);
                    }
                };
                copy_rows(r.dst_raw, r.raw);
                copy_rows(r.dst_values, r.values);
                if (r.finalize_mean && layer < r.sum.shape[0]) {
                    // The whole page is now complete, including a previously
                    // accepted partial prefix. Preserve finalize_mean_batch's
                    // +0 initialization, token order and FP32 additions. MTP
                    // layers beyond the scoring prefix are copied, not scored.
                    auto *out = reinterpret_cast<float*>(
                        static_cast<char*>(r.sum.ptr) + layer*r.sum.strides[0]);
                    const ssize_t channels = r.sum.shape[1]*r.sum.shape[2];
                    const bool fp8 = r.raw.shape[3] == r.values.shape[3];
                    std::fill(out, out+channels, 0.f);
                    for (ssize_t token=0; token<r.dst_raw.shape[1]; ++token) {
                        const char *row = static_cast<const char*>(r.dst_raw.ptr) +
                            layer*r.dst_raw.strides[0] + token*r.dst_raw.strides[1];
                        if (fp8) {
                            for (ssize_t channel=0; channel<channels; ++channel)
                                out[channel] += fp8_as_float(row+channel);
                        } else {
                            for (ssize_t channel=0; channel<channels; ++channel)
                                out[channel] += bf16_as_float(row+channel*2);
                        }
                    }
                }
            }
            // The caller finalizes this completed page from its immutable raw
            // bytes before committing history. None without this explicit flag
            // retains the original small-decode fused-reduction semantics.
            if (r.copy_only) return;
            auto *sum = static_cast<float *>(r.sum.ptr);
            if (r.delta.ptr) {
                const auto *delta = static_cast<const float *>(r.delta.ptr);
                for (ssize_t i = 0; i < r.sum.size; ++i) sum[i] += delta[i];
            } else {
                // Qualified BF16 decode rows, retaining ATen's FP32 outer
                // reduction order. Larger prefixes use the reference Torch sum.
                if (r.raw.shape[3]==r.values.shape[3]) {
                    switch (r.count) {
                        case 1: add_decode_mean<1,true>(r.raw,r.sum,r.begin); break;
                        case 2: add_decode_mean<2,true>(r.raw,r.sum,r.begin); break;
                        case 3: add_decode_mean<3,true>(r.raw,r.sum,r.begin); break;
                        case 4: add_decode_mean<4,true>(r.raw,r.sum,r.begin); break;
                    }
                } else switch (r.count) {
                    case 1: add_decode_mean<1>(r.raw, r.sum, r.begin); break;
                    case 2: add_decode_mean<2>(r.raw, r.sum, r.begin); break;
                    case 3: add_decode_mean<3>(r.raw, r.sum, r.begin); break;
                    case 4: add_decode_mean<4>(r.raw, r.sum, r.begin); break;
                }
            }
        };
        if (parallel) {
            kvmem_parallel::for_rows(rows.size(), scatter_row);
        } else {
            for (size_t index = 0; index < rows.size(); ++index) scatter_row(index);
        }
        }
        for (const auto &r : rows) finalized_records += r.finalize_mean;
        // Old 9/10-field callers keep the original None result. The explicit
        // fused descriptor reports actual execution, never silent fallback.
        for (const auto &job : jobs) if (job.size() == 11)
            return py::int_(finalized_records);
        return py::none();
    });

    m.def("pack_record_views", [](const std::vector<py::buffer> &sources) {
        // Sealed authority tensors stay owned by these exports. Copying their
        // immutable contents on an archive worker must not hold the engine GIL.
        std::vector<py::buffer_info> views;
        size_t total = 0;
        for (const auto &source : sources) {
            auto view = source.request();
            if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1 ||
                    static_cast<size_t>(view.size) > static_cast<size_t>(PY_SSIZE_T_MAX) - total)
                throw std::invalid_argument("Record pack requires bounded contiguous byte views");
            total += static_cast<size_t>(view.size);
            views.push_back(std::move(view));
        }
        if (!total) throw std::invalid_argument("Record pack requires nonempty bytes");
        PyObject *allocated = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(total));
        if (!allocated) throw py::error_already_set();
        auto result = py::reinterpret_steal<py::bytes>(allocated);
        char *destination = PyBytes_AS_STRING(result.ptr());
        {
            py::gil_scoped_release release;
            for (const auto &view : views) {
                std::memcpy(destination, view.ptr, static_cast<size_t>(view.size));
                destination += view.size;
            }
        }
        return result;
    });
    py::class_<vllm_kvmem::PrivateRecordPack>(m, "PrivateRecordPack")
        .def(py::init<const std::vector<py::buffer> &, size_t, size_t>(),
             py::arg("sources"), py::arg("record_bytes"), py::arg("max_payload_bytes"))
        .def("copy_range", &vllm_kvmem::PrivateRecordPack::copy_range)
        .def("finish", &vllm_kvmem::PrivateRecordPack::finish)
        .def("abort", &vllm_kvmem::PrivateRecordPack::abort);
    m.def("pack_record_pairs", [](const std::vector<py::buffer> &sources, size_t record_bytes) {
        // Independent immutable records, without a whole-batch slab or a
        // second copy. Validate and allocate under the GIL, copy the complete
        // admitted batch under one GIL release, then publish in the caller.
        if (sources.empty() || sources.size()%2 || !record_bytes ||
            record_bytes>static_cast<size_t>(PY_SSIZE_T_MAX))
            throw std::invalid_argument("Invalid CPU record pair extent");
        std::vector<py::buffer_info> views;
        views.reserve(sources.size());
        size_t pair_bytes=0;
        for (const auto &source : sources) {
            auto view=source.request();
            if (view.ndim!=1 || view.itemsize!=1 || view.strides[0]!=1 ||
                static_cast<size_t>(view.size)>record_bytes-pair_bytes)
                throw std::invalid_argument("Invalid CPU record pair view");
            pair_bytes+=static_cast<size_t>(view.size);
            views.push_back(std::move(view));
            if (views.size()%2==0) {
                if (pair_bytes!=record_bytes)
                    throw std::invalid_argument("CPU record pair size mismatch");
                pair_bytes=0;
            }
        }
        std::vector<py::bytes> records;
        std::vector<char *> destinations;
        records.reserve(sources.size()/2);
        destinations.reserve(sources.size()/2);
        for (size_t i=0;i<sources.size()/2;++i) {
            PyObject *allocated=PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(record_bytes));
            if (!allocated) throw py::error_already_set();
            records.push_back(py::reinterpret_steal<py::bytes>(allocated));
            destinations.push_back(PyBytes_AS_STRING(records.back().ptr()));
        }
        {
            py::gil_scoped_release release;
            for (size_t i=0;i<records.size();++i) {
                std::memcpy(destinations[i],views[2*i].ptr,views[2*i].size);
                std::memcpy(destinations[i]+views[2*i].size,views[2*i+1].ptr,views[2*i+1].size);
            }
        }
        return records;
    });
    m.def("copy_record_slices", [](py::buffer source, const std::vector<size_t> &offsets,
                                   size_t record_bytes) {
        auto view = source.request();
        if (!record_bytes || record_bytes > static_cast<size_t>(PY_SSIZE_T_MAX) ||
                view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1)
            throw std::invalid_argument("Invalid CPU record copy source or extent");
        const size_t extent = static_cast<size_t>(view.size);
        for (size_t offset : offsets)
            if (offset > extent || record_bytes > extent - offset)
                throw std::invalid_argument("CPU record copy outside source extent");
        std::vector<py::bytes> records;
        std::vector<char *> destinations;
        records.reserve(offsets.size());
        destinations.reserve(offsets.size());
        // Python allocation/refcounts require the GIL. These objects remain
        // private until every byte is initialized; no Python API is called in
        // the memcpy region. RAII restores the GIL before owners are destroyed.
        for (size_t i = 0; i < offsets.size(); ++i) {
            PyObject *allocated = PyBytes_FromStringAndSize(nullptr,
                static_cast<Py_ssize_t>(record_bytes));
            if (!allocated) throw py::error_already_set();
            records.push_back(py::reinterpret_steal<py::bytes>(allocated));
            destinations.push_back(PyBytes_AS_STRING(records.back().ptr()));
        }
        {
            py::gil_scoped_release release;
            const auto *input = static_cast<const char *>(view.ptr);
            for (size_t i = 0; i < offsets.size(); ++i)
                std::memcpy(destinations[i], input + offsets[i], record_bytes);
        }
        return records;
    });
    // Each source owns one immutable record. A caller may dispatch disjoint
    // ranges through a bounded persistent worker pool; retain Python owners and
    // the destination buffer export until all memcpy operations have completed.
    // Cache misses are left untouched for the authoritative tier read.
    m.def("gather_cached_records", [](py::buffer destination,
            const std::vector<py::object> &sources, size_t record_bytes,
            size_t begin, size_t end) {
        auto target = destination.request(true);
        if (!record_bytes || begin > end || end > sources.size() ||
                sources.size() > std::numeric_limits<size_t>::max() / record_bytes ||
                target.ndim != 1 || target.itemsize != 1 || target.strides[0] != 1 ||
                static_cast<size_t>(target.size) != sources.size() * record_bytes)
            throw std::invalid_argument("Invalid cached gather byte extent or range");
        std::vector<const char *> pointers;
        pointers.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            if (sources[i].is_none()) pointers.push_back(nullptr);
            else {
                if (!PyBytes_CheckExact(sources[i].ptr()) ||
                        static_cast<size_t>(PyBytes_GET_SIZE(sources[i].ptr())) != record_bytes)
                    throw std::invalid_argument("Cached gather sources must own exact immutable record bytes");
                pointers.push_back(PyBytes_AS_STRING(sources[i].ptr()));
            }
        }
        size_t copied = 0;
        {
            py::gil_scoped_release release;
            auto *out = static_cast<char *>(target.ptr);
            for (size_t i = begin; i < end; ++i) {
                if (const char *source = pointers[i - begin]) {
                    std::memcpy(out + i * record_bytes, source, record_bytes);
                    copied += record_bytes;
                }
            }
        }
        return copied;
    }, py::arg("destination"), py::arg("sources"), py::arg("record_bytes"),
       py::arg("begin"), py::arg("end"));
    m.def("gather_record_region", [](py::buffer destination,
            const std::vector<py::object> &sources, size_t record_bytes,
            size_t offset, size_t length, size_t begin, size_t end) {
        auto target = destination.request(true);
        if (!length || offset > record_bytes || length > record_bytes - offset ||
                begin > end || end > sources.size() ||
                sources.size() > static_cast<size_t>(PY_SSIZE_T_MAX) / length ||
                target.ndim != 1 || target.itemsize != 1 || target.strides[0] != 1 ||
                static_cast<size_t>(target.size) != sources.size() * length)
            throw std::invalid_argument("Invalid record region or destination extent");
        std::vector<py::buffer_info> exports;
        std::vector<const char *> pointers;
        for (size_t i = begin; i < end; ++i) {
            if (sources[i].is_none()) pointers.push_back(nullptr);
            else {
                auto view = py::reinterpret_borrow<py::buffer>(sources[i]).request();
                if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1 ||
                        static_cast<size_t>(view.size) != record_bytes)
                    throw std::invalid_argument("Region source must cover one complete record");
                pointers.push_back(static_cast<const char *>(view.ptr) + offset);
                exports.push_back(std::move(view));
            }
        }
        size_t copied = 0;
        // Opt-in KVMEM_NATIVE_GATHER_THREADS: disjoint record ranges in parallel
        // (raw pointers only; the exports above outlive every copy).
        const size_t threads = kvmem_parallel::gather_threads();
        {
            py::gil_scoped_release release;
            constexpr size_t kRows = 8;
            const size_t count = end - begin;
            if (threads > 1 && count > kRows) {
                std::atomic<size_t> total{0};
                kvmem_parallel::for_gather((count + kRows - 1) / kRows, [&](size_t task) {
                    const size_t lo = begin + task * kRows, hi = std::min(end, lo + kRows);
                    size_t local = 0;
                    for (size_t i = lo; i < hi; ++i) {
                        if (const char *source = pointers[i - begin]) {
                            std::memcpy(static_cast<char *>(target.ptr) + i * length, source, length);
                            local += length;
                        }
                    }
                    total.fetch_add(local);
                });
                copied = total.load();
            } else {
                for (size_t i = begin; i < end; ++i) {
                    if (const char *source = pointers[i - begin]) {
                        std::memcpy(static_cast<char *>(target.ptr) + i * length, source, length);
                        copied += length;
                    }
                }
            }
        }
        return copied;
    });
    // Experimental paired Query gather. Keep the original region entry point
    // untouched: this is selected explicitly by one archive read, never by a
    // process-wide dispatch change. The owning source vector and buffer exports
    // outlive both copies, including while the GIL is released.
    m.def("gather_record_regions", [](py::buffer destination,
            const std::vector<py::object> &sources, size_t record_bytes,
            const std::vector<std::pair<size_t, size_t>> &regions,
            size_t begin, size_t end) {
        using Clock = std::chrono::steady_clock;
        const auto started = Clock::now();
        auto target = destination.request(true);
        if (!record_bytes || regions.size() != 2 || begin > end || end > sources.size())
            throw std::invalid_argument("Invalid paired record regions or source range");
        size_t row_bytes = 0;
        for (const auto &[offset, length] : regions) {
            if (!length || offset > record_bytes || length > record_bytes - offset ||
                    length > static_cast<size_t>(PY_SSIZE_T_MAX) - row_bytes)
                throw std::invalid_argument("Invalid paired record region extent");
            row_bytes += length;
        }
        if (sources.size() > static_cast<size_t>(PY_SSIZE_T_MAX) / row_bytes ||
                target.ndim != 1 || target.itemsize != 1 || target.strides[0] != 1 ||
                static_cast<size_t>(target.size) != sources.size() * row_bytes)
            throw std::invalid_argument("Invalid paired record destination extent");
        std::vector<py::buffer_info> exports;
        std::vector<const char *> pointers;
        exports.reserve(end - begin);
        pointers.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            if (sources[i].is_none()) {
                pointers.push_back(nullptr);
            } else if (PyBytes_CheckExact(sources[i].ptr())) {
                if (static_cast<size_t>(PyBytes_GET_SIZE(sources[i].ptr())) != record_bytes)
                    throw std::invalid_argument("Paired gather source must cover one complete record");
                pointers.push_back(PyBytes_AS_STRING(sources[i].ptr()));
            } else {
                auto view = py::reinterpret_borrow<py::buffer>(sources[i]).request();
                if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1 ||
                        static_cast<size_t>(view.size) != record_bytes)
                    throw std::invalid_argument("Paired gather source must cover one complete record");
                pointers.push_back(static_cast<const char *>(view.ptr));
                exports.push_back(std::move(view));
            }
        }
        const auto prepared = Clock::now();
        size_t copied = 0;
        Clock::time_point copy_started, copy_finished;
        {
            py::gil_scoped_release release;
            copy_started = Clock::now();
            size_t cursor = 0;
            for (const auto &[offset, length] : regions) {
                for (size_t i = begin; i < end; ++i) {
                    if (const char *source = pointers[i - begin]) {
                        std::memcpy(static_cast<char *>(target.ptr) + cursor + i * length,
                                    source + offset, length);
                        copied += length;
                    }
                }
                cursor += sources.size() * length;
            }
            copy_finished = Clock::now();
        }
        const auto returned = Clock::now();
        const auto ns = [](auto duration) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        };
        py::dict result;
        result["copied_bytes"] = copied;
        result["source_prepare_ns"] = ns(prepared - started);
        result["copy_ns"] = ns(copy_finished - copy_started);
        result["gil_release_ns"] = ns(copy_started - prepared);
        result["gil_reacquire_ns"] = ns(returned - copy_finished);
        result["total_ns"] = ns(returned - started);
        return result;
    }, py::arg("destination"), py::arg("sources"), py::arg("record_bytes"),
       py::arg("regions"), py::arg("begin"), py::arg("end"));
    py::class_<KvMemGcConfig>(m, "GcPolicy")
        .def(py::init<>())
#define GCP(x) .def_readwrite(#x, &KvMemGcConfig::x)
        GCP(max_live_blocks) GCP(half_life_steps) GCP(grace_steps)
        GCP(candidate_steps) GCP(low_observations) GCP(scan_blocks) GCP(batch_blocks)
        GCP(poll_ms) GCP(worker_cpu) GCP(shadow) GCP(trace)
#undef GCP
        ;
    py::enum_<KvMemRepositoryGc::Mode>(m, "GcMode")
        .value("OFF", KvMemRepositoryGc::Mode::Off)
        .value("SHADOW", KvMemRepositoryGc::Mode::Shadow)
        .value("DELETE", KvMemRepositoryGc::Mode::Delete);
    py::class_<KvMemRepositoryGc::Config>(m, "RepositoryGcConfig")
        .def(py::init<>())
#define GCC(x) .def_readwrite(#x, &KvMemRepositoryGc::Config::x)
        GCC(policy) GCC(ready_low) GCC(ready_high) GCC(recent_tokens)
        GCC(candidate_ingest_tokens) GCC(tick_budget_us) GCC(mode) GCC(background)
#undef GCC
        ;
    py::class_<RepositoryGcBridge, std::shared_ptr<RepositoryGcBridge>>(m, "RepositoryGc")
        .def(py::init<KvMemRepositoryGc::Config, std::shared_ptr<KvMemRepository>, py::function>())
        .def("stop", &RepositoryGcBridge::stop)
        .def("poll", [](RepositoryGcBridge &gc) {
            py::gil_scoped_release no_gil;
            gc.native->poll();
        })
        .def("notify_progress", [](RepositoryGcBridge &gc) { gc.native->notify_progress(); })
        .def("begin_step", [](RepositoryGcBridge &gc) { return gc.native->begin_step(); })
        .def("set_history_end", [](RepositoryGcBridge &gc, uint64_t end) { gc.native->set_history_end(end); })
        .def("publish", [](RepositoryGcBridge &gc, const KvMemRepository::Lease &lease, uint32_t flags, bool used) {
            gc.native->publish(lease, flags, used);
        }, py::arg("lease"), py::arg("flags"), py::arg("used") = false)
        .def_property_readonly("last_error", &RepositoryGcBridge::last_error)
        .def_property_readonly("stats", [](RepositoryGcBridge &gc) {
            const auto s = gc.native->stats(); py::dict d;
#define GCS(x) d[#x] = s.x
            GCS(ticks); GCS(examined); GCS(retired); GCS(reclaimed); GCS(protected_blocks);
            GCS(deferred); GCS(failures); GCS(raced_updates); GCS(score_gaps); GCS(stale_scores);
            GCS(shadow_candidates); GCS(shadow_reuses); GCS(worker_ns); GCS(max_tick_ns);
            GCS(metadata_bytes); GCS(policy_fallback_retired); GCS(incomplete_backing); GCS(reader_conflicts);
#undef GCS
            return d;
        });
#define GCF(label, value) m.attr(label) = py::int_(uint32_t(value))
    GCF("GC_REGISTERED", KvMemGc::Registered); GCF("GC_SEALED", KvMemGc::Sealed);
    GCF("GC_GPU", KvMemGc::Gpu); GCF("GC_IO", KvMemGc::Io);
    GCF("GC_VALUE_BACKED", KvMemGc::ValueBacked); GCF("GC_RAW_BACKED", KvMemGc::RawBacked);
    GCF("GC_ANCHOR", KvMemGc::Anchor); GCF("GC_MANDATORY", KvMemGc::Mandatory);
    GCF("GC_CPU_CACHED", KvMemGc::CpuCached); GCF("GC_INDEXED", KvMemRepositoryGc::Indexed);
#undef GCF
    m.def("resolve_keep_bands", [](uint32_t block_tokens, uint32_t select_budget,
                                   int64_t sink_blocks, int64_t recent_blocks) {
        const auto bands = resolve_kvmem_keep_allocation(
            block_tokens, select_budget, sink_blocks, recent_blocks, -1, -1);
        return py::make_tuple(bands.sink_blocks, bands.recent_blocks);
    });
    py::class_<KvMemBlockHandle>(m, "Handle")
        .def(py::init<uint32_t, uint64_t>())
        .def_readonly("slot", &KvMemBlockHandle::slot)
        .def_readonly("generation", &KvMemBlockHandle::generation)
        .def("__eq__", &KvMemBlockHandle::operator==);
    py::class_<KvMemRepository::Record>(m, "Record")
        .def_readonly("handle", &KvMemRepository::Record::handle)
        .def_readonly("logical_id", &KvMemRepository::Record::logical_id)
        .def_readonly("token_begin", &KvMemRepository::Record::token_begin)
        .def_readonly("token_count", &KvMemRepository::Record::token_count);
    py::class_<KvMemRepository::Lease>(m, "Lease")
        .def("__bool__", [](const KvMemRepository::Lease &x) { return bool(x); })
        .def_property_readonly("handle", &KvMemRepository::Lease::handle)
        .def_property_readonly("record", &KvMemRepository::Lease::record,
                               py::return_value_policy::copy)
        .def("clone", &KvMemRepository::Lease::clone)
        .def("reset", &KvMemRepository::Lease::reset);
    py::class_<KvMemRepository::Reservation>(m, "Reservation")
        .def("__bool__", [](const KvMemRepository::Reservation &x) { return bool(x); })
        .def_property_readonly("record", &KvMemRepository::Reservation::record,
                               py::return_value_policy::copy)
        .def("lease", &KvMemRepository::Reservation::lease)
        .def("commit", &KvMemRepository::Reservation::commit)
        .def("cancel", &KvMemRepository::Reservation::cancel);
    py::enum_<KvMemRepository::ReclaimResult>(m, "ReclaimResult")
        .value("RECLAIMED", KvMemRepository::ReclaimResult::Reclaimed)
        .value("READERS", KvMemRepository::ReclaimResult::DeferredReaders)
        .value("RELEASE", KvMemRepository::ReclaimResult::DeferredRelease)
        .value("NOT_RETIRED", KvMemRepository::ReclaimResult::NotRetired)
        .value("STALE", KvMemRepository::ReclaimResult::StaleHandle)
        .value("BUSY", KvMemRepository::ReclaimResult::Busy);
    py::enum_<KvMemRepository::RetireResult>(m, "RetireResult")
        .value("COMMITTED", KvMemRepository::RetireResult::Committed)
        .value("ALREADY_RETIRED", KvMemRepository::RetireResult::AlreadyRetired)
        .value("STALE", KvMemRepository::RetireResult::StaleHandle)
        .value("RETRY", KvMemRepository::RetireResult::Retry);
    py::class_<KvMemRepository, std::shared_ptr<KvMemRepository>>(m, "Repository")
        .def(py::init([](uint32_t capacity, uint32_t block_tokens) {
            return std::make_shared<KvMemRepository>(KvMemRepository::Config{capacity, block_tokens});
        }))
        .def("reserve", &KvMemRepository::try_reserve)
        .def("pin", &KvMemRepository::try_pin)
        .def("pin_set", [](const KvMemRepository &r, const std::vector<KvMemBlockHandle> &handles) {
            LeaseSet set;
            set.leases.reserve(handles.size());
            for (const auto &handle : handles) set.leases.push_back(r.try_pin(handle));
            return set;
        })
        // [pin(h) for h in handles] in one call: identical leases, in order.
        .def("pin_many", [](const KvMemRepository &r, const std::vector<KvMemBlockHandle> &handles) {
            py::list leases;
            for (const auto &handle : handles)
                leases.append(py::cast(r.try_pin(handle), py::return_value_policy::move));
            return leases;
        })
        .def("pin_slot", &KvMemRepository::try_pin_slot)
        .def("retire", &KvMemRepository::retire)
        .def("retire_permanently", &KvMemRepository::retire_permanently)
        .def("retired_at", &KvMemRepository::retired_at)
        .def("reclaim", [](KvMemRepository &r, KvMemBlockHandle h, py::function release) {
            return r.try_reclaim(h, [&](const KvMemRepository::Record &record) {
                return release(py::cast(record)).cast<bool>();
            });
        })
        .def_property_readonly("ready_slots", &KvMemRepository::ready_slots)
        .def_property_readonly("stats", [](const KvMemRepository &r) {
            auto s = r.stats(); py::dict d;
#define STAT(x) d[#x] = s.x
            STAT(capacity); STAT(ready); STAT(writing); STAT(live); STAT(retired);
            STAT(releasing); STAT(cancelled); STAT(leases); STAT(reservations);
            STAT(reclaimed); STAT(metadata_bytes);
#undef STAT
            return d;
        });
    using PayloadArena = vllm_kvmem::HistoricalPayloadArena;
    py::class_<PayloadArena::ExternalCompletion, std::shared_ptr<PayloadArena::ExternalCompletion>>(m, "HistoricalIndexCompletion")
        .def("query", [](const PayloadArena::ExternalCompletion &f) {
            return f.done.load(std::memory_order_acquire);
        });
    py::class_<NativeIndexOwner, std::shared_ptr<NativeIndexOwner>>(m, "HistoricalIndexOwner", py::buffer_protocol())
        .def_buffer([](NativeIndexOwner &o) {
            return py::buffer_info(o.data, 1, py::format_descriptor<uint8_t>::format(), 1,
                {static_cast<py::ssize_t>(o.bytes)}, {1});
        })
        .def_property_readonly("completion", [](const NativeIndexOwner &o) { return o.owner->completion(); });
    m.def("flush_spill_ranges", [](const std::vector<int> &fds, const std::vector<uint64_t> &offsets,
                                   size_t record_bytes) {
        if (fds.size() != offsets.size()) throw std::invalid_argument("flush_spill_ranges needs one offset per fd");
        py::gil_scoped_release no_gil;
        vllm_kvmem::session_spill::flush_ranges(fds.data(), offsets.data(), fds.size(), record_bytes);
    });
    m.def("record_digest", [](py::buffer source) {
        auto view = source.request();
        if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1)
            throw std::invalid_argument("record_digest requires contiguous bytes");
        py::gil_scoped_release no_gil;
        return vllm_kvmem::session_spill::xxh64(view.ptr, static_cast<size_t>(view.size));
    });
    py::class_<PayloadArena::Handle>(m, "HistoricalPayloadHandle")
        .def(py::init<uint32_t, uint64_t>())
        .def_readonly("slot", &PayloadArena::Handle::slot)
        .def_readonly("generation", &PayloadArena::Handle::generation);
    py::class_<PayloadArena::Read>(m, "HistoricalPayloadRead", py::buffer_protocol())
        .def("__bool__", [](const PayloadArena::Read &r) { return bool(r); })
        .def("clone", &PayloadArena::Read::clone)
        .def_property_readonly("handle", &PayloadArena::Read::handle)
        .def_buffer([](PayloadArena::Read &r) {
            return py::buffer_info(const_cast<uint8_t *>(r.data()), 1,
                py::format_descriptor<uint8_t>::format(), 1,
                {static_cast<py::ssize_t>(r.size())}, {1}, true);
        });
    py::enum_<PayloadArena::RetireResult>(m, "HistoricalPayloadRetireResult")
        .value("COMMITTED", PayloadArena::RetireResult::Committed)
        .value("ALREADY_RETIRED", PayloadArena::RetireResult::AlreadyRetired)
        .value("STALE", PayloadArena::RetireResult::StaleHandle)
        .value("RETRY", PayloadArena::RetireResult::Retry)
        .value("SHARED", PayloadArena::RetireResult::Shared);
    py::class_<PayloadArena, std::shared_ptr<PayloadArena>>(m, "HistoricalPayloadArena")
        .def(py::init([](uint32_t capacity, size_t record_bytes, bool background,
                        uint32_t poll_ms, uint32_t scan_slots, uint32_t release_batch,
                        uint32_t pool_chunk_mib, uint32_t pool_ahead_records) {
            if (record_bytes > size_t(PY_SSIZE_T_MAX))
                throw std::invalid_argument("Historical CPU record exceeds buffer protocol limits");
            if (pool_ahead_records && !pool_chunk_mib)
                throw std::invalid_argument("Pre-faulted CPU records require a record pool");
            return std::make_shared<PayloadArena>(PayloadArena::Config{
                capacity, record_bytes, poll_ms, scan_slots, release_batch, background,
                pool_chunk_mib, pool_ahead_records});
        }), py::arg("capacity"), py::arg("record_bytes"), py::arg("background") = true,
            py::arg("poll_ms") = 10, py::arg("scan_slots") = 256, py::arg("release_batch") = 64,
            py::arg("pool_chunk_mib") = 0, py::arg("pool_ahead_records") = 0)
        .def("publish", [](PayloadArena &arena, PayloadArena::Handle handle, py::buffer source) {
            auto view = source.request();
            if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1)
                throw std::invalid_argument("Historical publication requires contiguous one-dimensional bytes");
            py::gil_scoped_release no_gil;
            arena.publish(handle, view.ptr, static_cast<size_t>(view.size));
        })
        .def("publish_segments", [](PayloadArena &arena, PayloadArena::Handle handle,
                                    const std::vector<py::buffer> &sources) {
            std::vector<py::buffer_info> exports;
            std::vector<PayloadArena::Segment> segments;
            exports.reserve(sources.size());
            segments.reserve(sources.size());
            for (const auto &source : sources) {
                auto view = source.request();
                if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1)
                    throw std::invalid_argument("Historical publication requires contiguous byte segments");
                segments.push_back({view.ptr, static_cast<size_t>(view.size)});
                exports.push_back(std::move(view));
            }
            py::gil_scoped_release no_gil;
            arena.publish_segments(handle, segments.data(), segments.size());
        })
        .def("acquire", &PayloadArena::acquire)
        .def("owns", &PayloadArena::owns)
        .def("_allocate_index", [](PayloadArena &arena, size_t bytes) {
            if (!bytes || bytes > size_t(PY_SSIZE_T_MAX))
                throw std::invalid_argument("Invalid native index allocation size");
            auto storage = std::make_unique<NativeIndexStorage>();
            auto result = std::make_shared<NativeIndexOwner>();
            result->bytes = bytes;
            result->owner = arena.retain_external(storage.get(), [](void *p) {
                delete static_cast<NativeIndexStorage *>(p);
            }, bytes);
            auto *admitted = storage.release();
            admitted->data.reset(new uint8_t[bytes]());
            result->data = admitted->data.get();
            return result;
        })
        .def("retire", &PayloadArena::retire)
        // Shared prefixes (docs/prefix-sharing.md, phase 2): owners of one payload.
        .def("share", &PayloadArena::share, py::arg("handle"), py::arg("pool") = false)
        .def("release_pool", &PayloadArena::release_pool)
        .def("retire_unshared", &PayloadArena::retire_unshared)
        .def("owners", [](const PayloadArena &arena, PayloadArena::Handle handle) {
            const auto owners = arena.owners(handle);
            return py::make_tuple(owners.total, owners.pool);
        })
        .def("reclaimed", &PayloadArena::reclaimed)
        // Opt-in session tiering (CPU TTL to SSD): drop / restore CPU bytes, same handle.
        .def("spill", [](PayloadArena &arena, PayloadArena::Handle handle) {
            py::gil_scoped_release no_gil;
            return arena.spill(handle);
        })
        .def("restore", [](PayloadArena &arena, PayloadArena::Handle handle, py::buffer source) {
            auto view = source.request();
            if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1)
                throw std::invalid_argument("Historical restore requires contiguous one-dimensional bytes");
            const PayloadArena::Segment segment{view.ptr, static_cast<size_t>(view.size)};
            py::gil_scoped_release no_gil;
            return arena.restore(handle, &segment, 1);
        })
        .def("spilled", &PayloadArena::spilled)
        // Batched session-tiering I/O (one GIL release per batch; see session_spill.hpp).
        .def("write_records", [](const PayloadArena &arena, const std::vector<PayloadArena::Handle> &handles,
                                 const std::vector<int> &fds, const std::vector<uint64_t> &offsets,
                                 bool drop_cache) {
            const size_t n = handles.size();
            if (fds.size() != n || offsets.size() != n)
                throw std::invalid_argument("write_records needs one fd and offset per record");
            std::vector<uint8_t> written(n);
            std::vector<uint64_t> digests(n);
            {
                py::gil_scoped_release no_gil;
                vllm_kvmem::session_spill::write_records(arena, handles.data(), fds.data(), offsets.data(), n,
                                                         written.data(), digests.data(), drop_cache);
            }
            std::vector<bool> ok(written.begin(), written.end());
            return py::make_tuple(ok, digests);
        }, py::arg("handles"), py::arg("fds"), py::arg("offsets"), py::arg("drop_cache") = false)
        .def("read_records", [](PayloadArena &arena, const std::vector<PayloadArena::Handle> &handles,
                                const std::vector<int> &fds, const std::vector<uint64_t> &offsets,
                                const std::vector<uint64_t> &digests, bool drop_cache) {
            const size_t n = handles.size();
            if (fds.size() != n || offsets.size() != n || digests.size() != n)
                throw std::invalid_argument("read_records needs one fd, offset and digest per record");
            std::vector<uint8_t> restored(n);
            {
                py::gil_scoped_release no_gil;
                vllm_kvmem::session_spill::read_records(arena, handles.data(), fds.data(), offsets.data(),
                                                        digests.data(), n, restored.data(), drop_cache);
            }
            return std::vector<bool>(restored.begin(), restored.end());
        }, py::arg("handles"), py::arg("fds"), py::arg("offsets"), py::arg("digests"),
           py::arg("drop_cache") = false)
        .def("spill_many", [](PayloadArena &arena, const std::vector<PayloadArena::Handle> &handles) {
            std::vector<bool> out(handles.size());
            py::gil_scoped_release no_gil;
            for (size_t i = 0; i < handles.size(); ++i) out[i] = arena.spill(handles[i]);
            return out;
        })
        .def("advance", [](PayloadArena &arena) {
            py::gil_scoped_release no_gil;
            return arena.advance();
        })
        .def("close", &PayloadArena::close)
        .def_property_readonly("stats", [](const PayloadArena &arena) {
            const auto s = arena.stats();
            py::dict result;
#define PAYLOAD_STAT(name) result[#name] = s.name
            PAYLOAD_STAT(used_slots); PAYLOAD_STAT(used_bytes); PAYLOAD_STAT(retired_bytes);
            PAYLOAD_STAT(readers); PAYLOAD_STAT(released_slots); PAYLOAD_STAT(released_bytes);
            PAYLOAD_STAT(last_release_thread);
            PAYLOAD_STAT(external_owners); PAYLOAD_STAT(external_bytes); PAYLOAD_STAT(external_released_bytes);
            PAYLOAD_STAT(pool_hits); PAYLOAD_STAT(pool_misses); PAYLOAD_STAT(pool_reserved_bytes);
            PAYLOAD_STAT(spilled_slots); PAYLOAD_STAT(spilled_bytes);
            PAYLOAD_STAT(shared_slots); PAYLOAD_STAT(pool_slots); PAYLOAD_STAT(unique_bytes);
#undef PAYLOAD_STAT
            return result;
        });
    using HistoryWorker = vllm_kvmem::HistoricalGcWorker;
    using HistoryScope = vllm_kvmem::HistoricalGcScope;
    using HistoryMode = vllm_kvmem::HistoricalGcMode;
    using HistoryCompletion = vllm_kvmem::HistoricalCompletion;
    py::enum_<HistoryMode>(m, "HistoricalGcMode")
        .value("OFF", HistoryMode::Off)
        .value("SHADOW", HistoryMode::Shadow)
        .value("DELETE", HistoryMode::Delete);
    using HistoryCommit = vllm_kvmem::HistoricalGcCommit;
    py::enum_<HistoryCommit>(m, "HistoricalGcCommit")
        .value("LOCAL", HistoryCommit::Local)
        .value("PROPOSE", HistoryCommit::Propose)
        .value("FOLLOW", HistoryCommit::Follow);
    py::class_<HistoryWorker::Config>(m, "HistoricalGcConfig")
        .def(py::init<>())
#define HCFG(name) .def_readwrite(#name, &HistoryWorker::Config::name)
        HCFG(max_scopes) HCFG(scope_capacity) HCFG(poll_ms) HCFG(ready_low) HCFG(ready_high)
        HCFG(cpu_free_low) HCFG(cpu_free_high) HCFG(scan_blocks) HCFG(delete_blocks)
        HCFG(mode) HCFG(commit) HCFG(policy) HCFG(keep_mandatory) HCFG(background)
#undef HCFG
        .def_property("policy_params", [](const HistoryWorker::Config &config) {
            return config.policy_params;
        }, [](HistoryWorker::Config &config, py::dict values) {
            vllm_kvmem::HistoricalPolicyParameters params;
            for (auto item : values) {
                if (!py::isinstance<py::str>(item.first))
                    throw std::invalid_argument("historical_gc.policy_params keys must be strings");
                const auto key = py::cast<std::string>(item.first);
                const auto value = item.second;
                // Check bool first and preserve scalar types. In particular,
                // integer 0/1 must not silently become a boolean parameter.
                if (PyBool_Check(value.ptr())) params[key] = py::cast<bool>(value);
                else if (PyLong_Check(value.ptr())) params[key] = py::cast<int64_t>(value);
                else if (PyFloat_Check(value.ptr())) params[key] = py::cast<double>(value);
                else if (PyUnicode_Check(value.ptr())) params[key] = py::cast<std::string>(value);
                else throw std::invalid_argument("historical_gc.policy_params values must be scalar");
            }
            config.policy_params = std::move(params);
        })
        ;
    py::class_<HistoryCompletion>(m, "HistoricalCompletion")
        .def_readonly("handle", &HistoryCompletion::handle)
        .def_readonly("cpu", &HistoryCompletion::cpu)
        .def_readonly("slot", &HistoryCompletion::slot);
    py::class_<vllm_kvmem::HistoricalFrameHold, std::shared_ptr<vllm_kvmem::HistoricalFrameHold>>(m, "HistoricalFrameHold");
    py::class_<HistoryScope, std::shared_ptr<HistoryScope>>(m, "HistoricalGcScope")
        .def("protect", &HistoryScope::protect)
        .def("protect_current", &HistoryScope::protect_current)
        // KVMEM_SESSION_FAST_TURN: protect / protect_current over a list of
        // exact (slot, generation) int tuples, without one Handle object per
        // record. NotImplemented for any other input (the caller then builds
        // Handles as before, with the original errors).
        .def("protect_pairs", [](HistoryScope &scope, const py::object &pairs) -> py::object {
            std::vector<KvMemBlockHandle> handles;
            if (!handles_from_pairs(pairs, handles)) return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            return py::cast(scope.protect(handles));
        })
        .def("protect_current_pairs", [](HistoryScope &scope, const py::object &pairs) -> py::object {
            std::vector<KvMemBlockHandle> handles;
            if (!handles_from_pairs(pairs, handles)) return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            return py::cast(scope.protect_current(handles));
        })
        .def("admit_read", [](HistoryScope &scope, KvMemBlockHandle handle) -> py::object {
            auto admission = scope.admit(handle);
            if (!admission.lease) return py::none();
            auto cpu = std::dynamic_pointer_cast<vllm_kvmem::HistoricalReadOwner>(admission.cpu);
            return py::make_tuple(std::move(admission.lease), cpu ? py::cast(cpu->read.clone()) : py::none());
        })
        .def("publish", &HistoryScope::publish, py::arg("lease"), py::arg("cpu"),
             py::arg("flags"), py::arg("use_seq") = 0)
        .def("update", &HistoryScope::update, py::arg("lease"), py::arg("flags"), py::arg("use_seq") = 0)
        .def("mark_shared", &HistoryScope::mark_shared, py::arg("lease"))
        .def("publish_thinking", &HistoryScope::publish_thinking, py::arg("lease"),
             py::arg("thinking_tokens"), py::arg("known_tokens"), py::arg("turn") = 0)
        .def("publish_turns", &HistoryScope::publish_turns, py::arg("completed"))
        .def("configure_features", &HistoryScope::configure_features,
             py::arg("layers"), py::arg("kv_heads"), py::arg("q_heads"), py::arg("dim"))
        .def("publish_features", [](HistoryScope &scope, const KvMemRepository::Lease &lease, py::buffer buffer) {
            const auto shape = scope.feature_geometry();
            if (!shape[0]) return false;
            auto info = historical_feature_buffer(buffer, {shape[0], shape[1], shape[3]});
            py::gil_scoped_release no_gil;
            return scope.publish_features(lease, static_cast<const float *>(info.ptr), info.size);
        }, py::arg("lease"), py::arg("buffer"))
        .def("begin_question", &HistoryScope::begin_question, py::arg("expected_tokens"))
        .def("append_question", [](HistoryScope &scope, uint64_t epoch, uint64_t offset,
                                    py::buffer buffer, uint32_t token_count) {
            const auto shape = scope.feature_geometry();
            if (!shape[0]) return false;
            auto info = historical_feature_buffer(buffer, {shape[0], token_count, shape[2], shape[3]});
            py::gil_scoped_release no_gil;
            return scope.append_question(epoch, offset, static_cast<const float *>(info.ptr), info.size, token_count);
        }, py::arg("epoch"), py::arg("offset"), py::arg("buffer"), py::arg("token_count"))
        .def("observe_query", [](HistoryScope &scope, uint64_t epoch, py::buffer buffer, uint32_t token_count) {
            const auto shape = scope.feature_geometry();
            if (!shape[0]) return false;
            auto info = historical_feature_buffer(buffer, {shape[0], token_count, shape[2], shape[3]});
            py::gil_scoped_release no_gil;
            return scope.observe_query(epoch, static_cast<const float *>(info.ptr), info.size, token_count);
        }, py::arg("epoch"), py::arg("buffer"), py::arg("token_count"))
        .def("invalidate_question", &HistoryScope::invalidate_question)
        .def_property_readonly("feature_stats", &HistoryScope::feature_stats)
        .def("used", &HistoryScope::used)
        .def("pause_policy", &HistoryScope::pause_policy)
        .def_property_readonly("policy_paused", &HistoryScope::policy_paused)
        .def("request_capacity", &HistoryScope::request_capacity, py::arg("ready_slots"),
             py::arg("retained_cpu_bytes") = UINT64_MAX)
        .def("completions", &HistoryScope::completions, py::arg("limit") = 64)
        .def("acknowledge", &HistoryScope::acknowledge)
        // Tensor-parallel synchronized commits (serving/tensor_parallel.gc_sync).
        .def("external_proposals", &HistoryScope::external_proposals,
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("external_proposal_ready", &HistoryScope::external_proposal_ready)
        .def("external_prepare", &HistoryScope::external_prepare, py::arg("handles"),
             py::call_guard<py::gil_scoped_release>())
        .def("external_commit", &HistoryScope::external_commit, py::arg("allowed"),
             py::call_guard<py::gil_scoped_release>())
        .def("external_release", &HistoryScope::external_release,
             py::call_guard<py::gil_scoped_release>())
        .def("close", &HistoryScope::close)
        .def_property_readonly("stats", [](const HistoryScope &scope) {
            const auto stats = scope.stats();
            py::dict result;
#define HSTAT(name) result[#name] = stats.name
            HSTAT(published); HSTAT(ticks); HSTAT(examined); HSTAT(decisions); HSTAT(retired);
            HSTAT(cpu_released); HSTAT(slots_released); HSTAT(pending_slots); HSTAT(pending_cpu_bytes);
            HSTAT(retries); HSTAT(stale); HSTAT(policy_failures); HSTAT(shadow_decisions);
            HSTAT(admission_attempts); HSTAT(admission_retries);
            HSTAT(redundancy_retired); HSTAT(lossy_retired); HSTAT(rejected_ineligible); HSTAT(cancelled_decisions);
            HSTAT(lru_fallback_retired); HSTAT(lru_representative_retired);
            HSTAT(thinking_classified); HSTAT(thinking_pure);
            HSTAT(shared_marked); HSTAT(shared_released);
            HSTAT(thinking_age_retired); HSTAT(thinking_age_ineligible); HSTAT(thinking_age_protected);
            HSTAT(thinking_age_passes); HSTAT(thinking_turns);
#undef HSTAT
            return result;
        });
    py::class_<HistoryWorker>(m, "HistoricalGcWorker")
        .def(py::init<HistoryWorker::Config, std::shared_ptr<PayloadArena>>())
        .def_property_readonly("storage_stats", [](const HistoryWorker &worker) {
            const auto stats = worker.storage_stats();
            py::dict result;
            result["allocations"]=stats.allocations; result["recycles"]=stats.recycles;
            result["pooled_scopes"]=stats.pooled_scopes; result["scope_capacity"]=stats.scope_capacity;
            return result;
        })
        .def("register_scope", [](HistoryWorker &worker, std::shared_ptr<KvMemRepository> repository) {
            py::gil_scoped_release no_gil;
            return worker.register_scope(std::move(repository));
        })
        .def("advance", [](HistoryWorker &worker) {
            py::gil_scoped_release no_gil;
            worker.advance();
        })
        .def("set_commit", &HistoryWorker::set_commit, py::arg("mode"))
        .def_property_readonly("commit", &HistoryWorker::commit)
        .def("close", &HistoryWorker::close);
    m.attr("HISTORY_SEALED") = py::int_(uint32_t(vllm_kvmem::HistoricalSealed));
    m.attr("HISTORY_MANDATORY") = py::int_(uint32_t(vllm_kvmem::HistoricalMandatory));
    m.attr("HISTORY_GPU") = py::int_(uint32_t(vllm_kvmem::HistoricalGpu));
    py::class_<KvMemStoreConfig>(m, "StoreConfig")
        .def(py::init<>())
#define CFG(x) .def_readwrite(#x, &KvMemStoreConfig::x)
        CFG(block_tokens) CFG(select_budget) CFG(prefill_budget) CFG(gen_budget)
        CFG(sink_blocks) CFG(recent_blocks) CFG(repository_blocks)
        CFG(immutable_source_k) CFG(immutable_refresh_abs_delta_tokens)
        CFG(mtp_enabled) CFG(action_fraction) CFG(action_combine)
#undef CFG
        ;
    py::class_<KvMemStorageSpan>(m, "StorageSpan")
        .def_readonly("history_begin", &KvMemStorageSpan::history_begin)
        .def_readonly("storage_begin", &KvMemStorageSpan::storage_begin)
        .def_readonly("token_count", &KvMemStorageSpan::token_count)
        .def_readonly("handle", &KvMemStorageSpan::handle);
    py::class_<OwnedRead>(m, "StorageRead")
        .def_property_readonly("spans", [](const OwnedRead &r) { return r.spans; });
    py::class_<KvMemRemap>(m, "Remap")
#define REMAP(x) .def_readonly(#x, &KvMemRemap::x)
        REMAP(block_id) REMAP(n_tokens) REMAP(from_base) REMAP(to_base)
        REMAP(skip) REMAP(raw_refresh) REMAP(working_k_resident)
#undef REMAP
        ;
    py::class_<KvMemPlan>(m, "Plan")
        .def("admitted_read", [](const KvMemPlan &plan, KvMemBlockHandle handle) -> py::object {
            if (!plan.repository_next_window) return py::none();
            for (size_t i = 0; i < plan.repository_next_window->size(); ++i) {
                const auto &lease = (*plan.repository_next_window)[i];
                if (lease.handle() != handle) continue;
                auto cpu = std::dynamic_pointer_cast<vllm_kvmem::HistoricalReadOwner>(plan.cpu_sources[i]);
                return py::make_tuple(lease.clone(), cpu ? py::cast(cpu->read.clone()) : py::none());
            }
            throw std::invalid_argument("Read is outside the admitted assembly window");
        })
        .def("release_cpu_sources", [](KvMemPlan &plan) {
            for (auto &source : plan.cpu_sources) source.reset();
        })
#define PLAN(x) .def_readonly(#x, &KvMemPlan::x)
        PLAN(stage_in) PLAN(stage_out) PLAN(remaps) PLAN(total_window_tokens)
        PLAN(selection_overlap_blocks) PLAN(gpu_reused_blocks)
        PLAN(retained_position_stable) PLAN(retained_position_moved)
#undef PLAN
        ;
    // Opt-in KVMEM_PRESSURE_NATIVE_PLAN_CHECK: the layer-zero single-assembly
    // checks of layerwise_pressure_efficiency and _validate_destinations, read
    // from the plan's C++ remaps instead of 1,152 converted Python objects.
    // Same checks, order and messages. Returns (skip_begins, rebuilt_tokens,
    // skipped_tokens, partial_tokens) for the unchanged AssemblyResult fields.
    m.def("check_pressure_plan", [](const KvMemPlan &plan, const std::vector<uint32_t> &slots,
                                     const std::vector<int64_t> &counts, const std::vector<int64_t> &begins,
                                     const std::vector<std::pair<int64_t, int64_t>> &layer_bounds,
                                     bool original_rope, bool rolling) {
        if (slots.size() != counts.size() || slots.size() != begins.size())
            throw std::invalid_argument("Invalid layer-zero selection arrays");
        std::unordered_map<uint32_t, size_t> by_slot;   // last wins, like a dict
        std::unordered_map<uint32_t, int64_t> positions;
        by_slot.reserve(slots.size()); positions.reserve(slots.size());
        int64_t cursor = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            by_slot[slots[i]] = i;
            positions[slots[i]] = cursor;
            cursor += counts[i];
        }
        const auto &remaps = plan.remaps;
        {
            std::unordered_map<uint32_t, bool> seen;
            seen.reserve(remaps.size());
            for (const auto &r : remaps) seen[r.block_id] = true;
            bool same = seen.size() == by_slot.size();
            if (same) for (const auto &item : by_slot) if (!seen.count(item.first)) { same = false; break; }
            if (same) for (const auto &r : remaps) if (r.to_base != positions[r.block_id]) { same = false; break; }
            if (!same) throw py::value_error("Layer-zero selection differs from the native assembly plan");
        }
        bool have_bounds = false;
        int64_t capacity = 0, rope = 0;
        for (const auto &r : remaps) {
            const size_t index = by_slot.at(r.block_id);
            if (counts[index] != static_cast<int64_t>(r.n_tokens) || r.to_base < 0)
                throw py::value_error("Selected generation/extent no longer matches authority");
            if (!have_bounds) {
                have_bounds = true;
                capacity = std::numeric_limits<int64_t>::max();
                rope = std::numeric_limits<int64_t>::max();
                for (const auto &b : layer_bounds) { capacity = std::min(capacity, b.first); rope = std::min(rope, b.second); }
            }
            const int64_t end = static_cast<int64_t>(r.to_base) + r.n_tokens;
            const int64_t rope_end = original_rope ? begins[index] + r.n_tokens : end;
            if (end > capacity || rope_end > rope) {
                for (const auto &b : layer_bounds) {
                    if (end > b.first) throw py::value_error("Selected working window exceeds native allocated pages");
                    if (rope_end > b.second) throw py::value_error("Selected position exceeds the model RoPE table");
                }
            }
        }
        py::list skip;
        int64_t rebuilt = 0, skipped = 0, partial = 0;
        for (const auto &r : remaps) {
            if (!rolling && r.skip) {
                skip.append(begins[by_slot.at(r.block_id)]);
                skipped += r.n_tokens;
            } else {
                rebuilt += r.n_tokens;
            }
            if (!r.skip && r.n_tokens != 32) partial += r.n_tokens;
        }
        return py::make_tuple(skip, rebuilt, skipped, partial);
    }, py::arg("plan"), py::arg("slots"), py::arg("counts"), py::arg("begins"), py::arg("layer_bounds"),
       py::arg("original_rope"), py::arg("rolling"));
    py::class_<KvMemStore::RepositoryAppend>(m, "Append")
        .def_property_readonly("spans", &KvMemStore::RepositoryAppend::spans)
        .def_property_readonly("begin", &KvMemStore::RepositoryAppend::begin)
        .def_property_readonly("end", &KvMemStore::RepositoryAppend::end)
        .def("commit", &KvMemStore::RepositoryAppend::commit)
        .def("storage_write", [](const KvMemStore::RepositoryAppend &a, uint64_t b, uint32_t n) {
            return OwnedRead(a.storage_write(b, n));
        })
        .def("discarded_page_ranges", &KvMemStore::RepositoryAppend::discarded_page_ranges);
    py::class_<OwnedScoreFrame>(m, "QueryScoreFrame")
        .def_readonly("score_mask", &OwnedScoreFrame::score_mask)
        .def_readonly("transport_mask", &OwnedScoreFrame::transport_mask)
        .def("close", &OwnedScoreFrame::close);
    py::class_<KvMemStore>(m, "Store")
        .def(py::init<KvMemStoreConfig, std::shared_ptr<KvMemRepository>>())
        .def("attach_history", [](KvMemStore &store, std::shared_ptr<HistoryScope> scope) {
            if (scope && store.repository() == nullptr)
                throw std::invalid_argument("Historical scope requires a repository");
            store.attach_history_observer(std::move(scope));
        })
        .def("attach_gc", [](KvMemStore &store, std::shared_ptr<RepositoryGcBridge> gc) {
            py::gil_scoped_release no_gil;
            store.attach_repository_gc(gc ? gc->native : nullptr);
        }, py::keep_alive<1, 2>())
        .def("gc_begin_step", &KvMemStore::gc_begin_step)
        .def("gc_end_step", &KvMemStore::gc_end_step)
        .def("gc_flags", &KvMemStore::set_repository_gc_flags)
        .def("working_resident", [](KvMemStore &store, KvMemBlockHandle handle, bool resident) {
            // The plugin owns archive locations; no fictitious native CPU/SSD
            // slot is installed. Durable raw/V/index flags are independent.
            store.set_block_tier(handle.slot, resident ? KvTier::GPU : KvTier::SSD,
                                 -1, -1, handle.generation);
        })
        .def("set_protect_current_window", &KvMemStore::set_protect_current_window)
        .def_property_readonly("protect_current_window", &KvMemStore::protect_current_window)
        // Prefix-image import (serving/prefix_images.py): install one committed
        // repository generation as the next whole history block. The Python
        // lease is moved into the store (it reads false afterwards). With
        // current-window protection (configured before the first install),
        // current_frame marks the block as part of the protected current frame.
        .def("register_block", [](KvMemStore &store, KvMemRepository::Lease &lease, uint32_t baked_pos,
                                  bool current_frame) {
            store.register_repository_block(std::move(lease), baked_pos, current_frame);
        }, py::arg("lease"), py::arg("baked_pos"), py::arg("current_frame") = true)
        .def("prepare_append", &KvMemStore::prepare_repository_append, py::keep_alive<0, 1>())
        .def("checkpoint", &KvMemStore::repository_checkpoint,
             py::arg("position"), py::arg("pin_working_set") = true)
        .def("checkpoint_valid", &KvMemStore::repository_checkpoint_valid)
        .def("current_frame_records", [](const KvMemStore &store) {
            // Diagnostics: the blocks of the protected current frame (historical
            // GC never retires them), as (slot, generation, history_begin).
            std::vector<std::tuple<uint32_t, uint64_t, uint64_t>> records;
            if (!store.protect_current_window()) return records;
            for (const auto &block : store.blocks()) {
                if (block.repository_generation && block.n_tokens && block.in_current_frame)
                    records.emplace_back(block.block_id, block.repository_generation, block.history_pos_start);
            }
            return records;
        })
        .def("working_records", [](const KvMemStore &store) {
            // Value tuples only: no block/Store views survive into a checkpoint.
            std::vector<std::tuple<uint32_t, uint64_t, uint64_t, uint32_t, int64_t>> records;
            const auto &blocks = store.blocks();
            for (size_t i = 0, end = store.repository_scan_end(); i < end; ++i) {
                const auto &block = blocks[i];
                if (block.repository_generation && block.n_tokens &&
                        (block.in_working_set || block.tier == KvTier::GPU))
                    records.emplace_back(block.block_id, block.repository_generation,
                        block.history_pos_start, block.n_tokens, block.baked_pos);
            }
            std::sort(records.begin(), records.end(), [](const auto &a, const auto &b) {
                return std::get<4>(a) < std::get<4>(b);
            });
            return records;
        })
        .def("action_window_segments", [](const KvMemStore &store, int64_t history_begin,
                int64_t local_begin, int64_t count, int64_t block_tokens) {
            const auto segments = kvmem_action::window_segments(
                store, history_begin, local_begin, count, block_tokens);
            py::array_t<int32_t> starts(segments.starts.size()), counts(segments.counts.size());
            if (!segments.starts.empty()) {
                std::memcpy(starts.mutable_data(), segments.starts.data(), segments.starts.size()*sizeof(int32_t));
                std::memcpy(counts.mutable_data(), segments.counts.data(), segments.counts.size()*sizeof(int32_t));
            }
            // History remains Python ints: Action/PrefillHistory dict keys and
            // snapshot JSON must never silently become NumPy scalar objects.
            return py::make_tuple(std::move(starts), std::move(counts), py::cast(segments.history));
        }, py::arg("history_begin"), py::arg("local_begin"), py::arg("count"), py::arg("block_tokens"))
        // Opt-in KVMEM_NATIVE_WINDOW_FRAME: compare the canonical action window
        // with a frozen decode frame's prefix without building Python rows.
        .def("action_window_matches_frame", [](const KvMemStore &store, int64_t history_begin,
                int64_t local_begin, int64_t count, int64_t block_tokens,
                py::array_t<int64_t, py::array::c_style> starts,
                py::array_t<int64_t, py::array::c_style> counts,
                py::array_t<int64_t, py::array::c_style> history) {
            if (starts.ndim() != 1 || counts.ndim() != 1 || history.ndim() != 1 ||
                    counts.shape(0) != starts.shape(0) || history.shape(0) != starts.shape(0))
                return false;
            return kvmem_action::window_matches_frame(store, history_begin, local_begin, count,
                block_tokens, starts.data(), counts.data(), history.data(),
                static_cast<size_t>(starts.shape(0)));
        }, py::arg("history_begin"), py::arg("local_begin"), py::arg("count"), py::arg("block_tokens"),
           py::arg("starts"), py::arg("counts"), py::arg("history"))
        .def("prepare_rewind", &KvMemStore::prepare_repository_rewind, py::keep_alive<0, 1>())
        .def("read", [](const KvMemStore &s, uint64_t b, uint32_t n) {
            return OwnedRead(s.repository_storage_read(b, n));
        })
        .def("handle", &KvMemStore::repository_handle)
        // [handle(slot).generation for slot in slots] in one call.
        .def("generations", [](const KvMemStore &store, const std::vector<uint32_t> &slots) {
            std::vector<uint64_t> out;
            out.reserve(slots.size());
            for (const auto slot : slots) out.push_back(store.repository_handle(slot).generation);
            return out;
        })
        .def_property_readonly("history_end", &KvMemStore::history_end)
        .def_property_readonly("runtime_select_budget", &KvMemStore::runtime_select_budget_tokens)
        .def("query_score_frame", [](KvMemStore &store,
                const std::vector<KvMemBlockHandle> &handles, const std::vector<uint32_t> &counts,
                uint64_t boundary, const std::vector<uint32_t> &mandatory, uint32_t budget) {
            if (handles.size() != counts.size() || boundary > store.history_end())
                throw std::invalid_argument("Invalid Query score index shape or accepted boundary");
            ScopedSelectionBudget scoped(store, budget);
            std::vector<KvMemIndexStamp> index(store.block_count());
            std::vector<uint8_t> seen(store.block_count(), 0);
            for (size_t i=0; i<handles.size(); ++i) {
                auto h=handles[i];
                if (h.slot >= index.size() || !h.generation ||
                        store.repository_handle(h.slot) != h || seen[h.slot] || !counts[i])
                    throw std::invalid_argument("Stale, duplicate or invalid Query score index");
                seen[h.slot]=1;
                index[h.slot]={h,counts[i]};
            }
            for (auto slot : mandatory)
                if (slot >= seen.size() || !seen[slot])
                    throw std::invalid_argument("Mandatory block outside Query score index");
            auto result=std::make_unique<OwnedScoreFrame>();
            result->frame=store.repository_score_frame(index,boundary,true);
            const auto fixed=store.fixed_selection_blocks(mandatory);
            std::vector<uint8_t> kept(store.block_count(),0);
            for (auto slot:fixed) kept[slot]=1;
            for (auto h:handles) {
                result->score_mask.push_back(result->frame.enabled[h.slot]!=0);
                result->transport_mask.push_back(!kept[h.slot]);
            }
            return result;
        }, py::arg("handles"), py::arg("counts"), py::arg("boundary"),
           py::arg("mandatory")=std::vector<uint32_t>{}, py::arg("budget")=0)
        .def("fixed_selection_blocks", &KvMemStore::fixed_selection_blocks,
             py::arg("mandatory"), py::arg("pressure")=false, py::arg("budget")=0)
        .def("scores", &KvMemStore::set_repository_retrieval_scores)
        .def("set_runtime_select_budget", &KvMemStore::set_runtime_select_budget)
        .def("set_action_scores", [](KvMemStore &s, const std::unordered_map<uint64_t,double> &scores) {
            for (const auto &[key,value]:scores)
                if (!std::isfinite(value) || value<0) throw std::invalid_argument("Invalid action score");
            std::vector<double> values(s.block_count(),0);
            for (const auto &b:s.blocks()) if (s.is_live(b.block_id)) {
                const auto it=scores.find(b.history_pos_start);
                if (it!=scores.end()) values[b.block_id]=it->second;
            }
            s.update_action_scores(values,0);
        })
        .def("pick", [](const KvMemStore &s, const std::vector<uint32_t> &mandatory) {
            return s.pick_topk_blocks(mandatory);
        })
        .def("pick_groups", [](const KvMemStore &s,
                const std::vector<std::pair<uint64_t, uint64_t>> &groups,
                const std::vector<double> &scores, const std::vector<uint32_t> &mandatory) {
            if (groups.size() != scores.size())
                throw std::invalid_argument("Group spans and scores must have equal lengths");
            for (size_t i = 0; i < groups.size(); ++i) {
                if (groups[i].second <= groups[i].first || groups[i].second > s.history_end())
                    throw std::invalid_argument("Group span is outside accepted logical history");
                if (!std::isfinite(scores[i]))
                    throw std::invalid_argument("Group scores must be finite");
            }
            // Logical-span lookup validates live generations and skips an
            // incomplete group as a whole. Actual execution admission remains
            // in Store.select; neither scoring nor graph masks grant access.
            return s.pick_repository_semantic_groups(groups, scores, mandatory);
        }, py::arg("groups"), py::arg("group_scores"),
           py::arg("mandatory") = std::vector<uint32_t>{})
        .def("pick_with_prefill", [](const KvMemStore &s, const std::vector<uint32_t> &mandatory,
                                    const KvMemStore::PrefillSelectionScores &scores) {
            for (const auto &[key,value]:scores)
                if (!std::isfinite(value) || value<0) throw std::invalid_argument("Invalid prefill score");
            return s.pick_topk_blocks(mandatory, scores);
        })
        .def("pick_pressure", [](const KvMemStore &s, const std::vector<uint32_t> &mandatory,
                                 const KvMemStore::PressureActionScores &scores, uint32_t budget) {
            return s.pick_prefill_pressure_blocks(mandatory, scores, budget);
        }, py::arg("mandatory"), py::arg("scores"), py::arg("budget") = 0)
        // Opt-in KVMEM_PRESSURE_NATIVE_LAYER_PICK: one lazily selected layer of
        // LayerwisePrefillPressure.prepare.select_layer (fast_prepare + native
        // score lookups) in one call, in the original order: the candidate loop
        // with its checks and messages, then preview_selection's READY check
        // (``require``), its empty retrieval-score frame and the pressure pick,
        // then the layer checks and the archive-order sort. Returns None before
        // any side effect when an input is not in the expected shape, so the
        // caller runs the original Python path instead.
        .def("pressure_layer_pick", [](KvMemStore &s, const py::list &mapping, const py::dict &by_begin,
                const py::list &scores, const std::vector<uint32_t> &mandatory, uint32_t budget,
                const py::dict &record_positions, const py::list &records,
                const py::object &require) -> py::object {
            const Py_ssize_t count = PyList_GET_SIZE(mapping.ptr());
            if (PyList_GET_SIZE(scores.ptr()) != count) return py::none();
            for (Py_ssize_t i = 0; i < count; ++i) {
                PyObject *row = PyList_GET_ITEM(mapping.ptr(), i);
                if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 5) return py::none();
                if (!PyFloat_CheckExact(PyList_GET_ITEM(scores.ptr(), i))) return py::none();
            }
            py::list candidates, candidate_slots;
            KvMemStore::PressureActionScores supplied;
            supplied.reserve(static_cast<size_t>(count));
            std::unordered_set<uint32_t> allowed(mandatory.begin(), mandatory.end());
            bool positive = false;
            for (Py_ssize_t i = 0; i < count; ++i) {
                PyObject *row = PyList_GET_ITEM(mapping.ptr(), i);
                PyObject *begin = PyTuple_GET_ITEM(row, 2), *n = PyTuple_GET_ITEM(row, 3);
                PyObject *entry = PyDict_GetItemWithError(by_begin.ptr(), begin);
                if (!entry && PyErr_Occurred()) throw py::error_already_set();
                bool absent = entry == nullptr || entry == Py_None;
                if (!absent) {
                    if (!PyTuple_CheckExact(entry) || PyTuple_GET_SIZE(entry) != 2) return py::none();
                    py::object record_count = py::handle(PyTuple_GET_ITEM(entry, 0)).attr("count");
                    const int differs = PyObject_RichCompareBool(record_count.ptr(), n, Py_NE);
                    if (differs < 0) throw py::error_already_set();
                    absent = differs != 0;
                }
                if (absent)
                    throw std::runtime_error("Layer pressure candidate is absent from accepted immutable history");
                PyObject *record = PyTuple_GET_ITEM(entry, 0), *slot = PyTuple_GET_ITEM(entry, 1);
                const double score = PyFloat_AS_DOUBLE(PyList_GET_ITEM(scores.ptr(), i));
                if (!std::isfinite(score) || score < 0)
                    throw std::runtime_error("Invalid layer cumulative attention score");
                positive = positive || score > 0;
                candidates.append(py::handle(record));
                candidate_slots.append(py::handle(slot));
                const auto slot_id = py::cast<uint32_t>(py::handle(slot));
                supplied.emplace_back(slot_id, score);
                allowed.insert(slot_id);
            }
            if (!positive)
                throw std::runtime_error("Layer pressure has no accepted attention; refusing a recency fallback");
            require();                                    // preview_selection: _require(Phase.READY)
            s.set_repository_retrieval_scores({}, {});    // preview_selection: store.scores([], [])
            const auto picked_ids = s.pick_prefill_pressure_blocks(mandatory, supplied, budget);
            std::unordered_set<uint32_t> slots(picked_ids.begin(), picked_ids.end());
            bool violated = 32ull * slots.size() > budget;
            for (const auto id : mandatory) violated = violated || !slots.count(id);
            for (const auto id : slots) violated = violated || !allowed.count(id);
            if (violated)
                throw std::runtime_error("Layer pressure violated candidates, mandatory blocks or Active budget");
            std::vector<std::pair<Py_ssize_t, uint32_t>> order;
            order.reserve(slots.size());
            for (const auto id : slots) {
                py::object key = py::int_(id);
                PyObject *position = PyDict_GetItemWithError(record_positions.ptr(), key.ptr());
                if (!position) {
                    if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, key.ptr());
                    throw py::error_already_set();
                }
                const Py_ssize_t index = PyLong_AsSsize_t(position);
                if (index == -1 && PyErr_Occurred()) throw py::error_already_set();
                order.emplace_back(index, id);
            }
            std::sort(order.begin(), order.end());
            py::list ordered_slots, picked;
            for (const auto &[index, id] : order) {
                ordered_slots.append(py::int_(id));
                if (index < 0 || index >= PyList_GET_SIZE(records.ptr())) {
                    PyErr_SetString(PyExc_IndexError, "list index out of range");
                    throw py::error_already_set();
                }
                picked.append(py::handle(PyList_GET_ITEM(records.ptr(), index)));
            }
            return py::make_tuple(candidates, candidate_slots, ordered_slots, picked);
        }, py::arg("mapping"), py::arg("by_begin"), py::arg("scores"), py::arg("mandatory"),
           py::arg("budget"), py::arg("record_positions"), py::arg("records"), py::arg("require"))
        .def("pick_pressure_batch", [](const KvMemStore &s, const std::vector<uint32_t> &mandatory,
                const std::vector<std::vector<uint32_t>> &layer_slots,
                const std::vector<std::vector<double>> &layer_scores, uint32_t budget) {
            if (layer_slots.size() != layer_scores.size())
                throw std::invalid_argument("Pressure slot and score layer counts differ");
            std::vector<KvMemStore::PressureActionScores> pairs(layer_slots.size());
            for (size_t layer = 0; layer < pairs.size(); ++layer) {
                if (layer_slots[layer].size() != layer_scores[layer].size())
                    throw std::invalid_argument("Pressure slot and score row counts differ");
                pairs[layer].reserve(layer_slots[layer].size());
                for (size_t i = 0; i < layer_slots[layer].size(); ++i)
                    pairs[layer].emplace_back(layer_slots[layer][i], layer_scores[layer][i]);
            }
            return s.pick_prefill_pressure_batch(mandatory, pairs, budget);
        }, py::arg("mandatory"), py::arg("layer_slots"), py::arg("layer_scores"), py::arg("budget") = 0)
        .def("select", &KvMemStore::set_repository_selection,
             py::arg("handles"), py::arg("force_raw_refresh") = false)
        // select([handle(slot) for slot in slots]) without per-slot Python handles.
        .def("select_slots", [](KvMemStore &store, const std::vector<uint32_t> &slots) {
            std::vector<KvMemBlockHandle> handles;
            handles.reserve(slots.size());
            for (const auto slot : slots) handles.push_back(store.repository_handle(slot));
            return store.set_repository_selection(handles, false);
        }, py::arg("slots"))
        .def("restore_selection", &KvMemStore::restore_repository_selection,
             py::arg("handles"))
        // Experimental host adapter; serving continues to use RequestMemory's
        // original Python restore path until separately qualified and enabled.
        .def("restore_frame", &kvmem_restore::restore_frame,
             py::arg("prefix"), py::arg("records"))
        .def("validate_frame", [](KvMemStore &store,
                const std::shared_ptr<KvMemStore::RepositoryCheckpoint> &prefix,
                const std::vector<kvmem_restore::FrameRecord> &records) {
            // Preserve the single producer/GIL contract used by restore_frame.
            // The returned read leases die here, after all checks; the caller
            // continues holding the checkpoint and historical policy protection.
            kvmem_restore::validate_frame(store, prefix, records);
        }, py::arg("prefix"), py::arg("records"))
        .def("set_baked", &KvMemStore::set_block_baked_pos,
             py::arg("slot"), py::arg("position"), py::arg("generation"));

    // RequestMemory._publish_selection's mapping loop in one call
    // (KVMEM_NATIVE_PUBLISH_SELECTION): the same store calls in the same order
    // as the Python loop (stage_out leaves the working set, then every remap
    // enters it and gets its baked position), without converting the plan's
    // vectors to Python objects. Holds the GIL like the Python loop it replaces.
    m.def("publish_selection", [](KvMemStore &store, const KvMemPlan &plan) {
        for (const auto slot : plan.stage_out) {
            const auto handle = store.repository_handle(slot);
            store.set_block_tier(handle.slot, KvTier::SSD, -1, -1, handle.generation);
        }
        for (const auto &remap : plan.remaps) {
            const auto handle = store.repository_handle(remap.block_id);
            store.set_block_tier(handle.slot, KvTier::GPU, -1, -1, handle.generation);
            store.set_block_baked_pos(handle.slot, remap.to_base, handle.generation);
        }
    }, py::arg("store"), py::arg("plan"));

    py::class_<KvMemStore::RepositoryCheckpoint, std::shared_ptr<KvMemStore::RepositoryCheckpoint>>(
            m, "RepositoryCheckpoint")
        .def_property_readonly("position", &KvMemStore::RepositoryCheckpoint::position)
        .def_property_readonly("valid", &KvMemStore::RepositoryCheckpoint::valid);
    py::class_<KvMemStore::RepositoryRewind>(m, "RepositoryRewind")
        .def_property_readonly("position", &KvMemStore::RepositoryRewind::position)
        .def_property_readonly("discarded_spans", &KvMemStore::RepositoryRewind::discarded_spans)
        .def("commit", &KvMemStore::RepositoryRewind::commit);

    py::class_<NvmeAllocation>(m, "NvmeAllocation")
        .def_readonly("block_id", &NvmeAllocation::block_id)
        .def_readonly("slot", &NvmeAllocation::slot)
        .def_readonly("generation", &NvmeAllocation::generation);
    py::class_<NvmeIoSpan>(m, "IoSpan")
        .def(py::init<int32_t, uint64_t, uint64_t>());
    py::class_<NvmeKvTier>(m, "NvmeTier")
        .def(py::init([](const std::string &dir, uint64_t total, uint64_t slot, bool preallocate) {
            if (slot == 0 || total < slot || total / slot > UINT32_MAX)
                throw std::invalid_argument("Invalid NVMe arena capacity");
            py::gil_scoped_release release;
            NvmeKvTierConfig cfg;
            cfg.dir = dir; cfg.total_bytes = total; cfg.slot_bytes = slot;
            cfg.drop_page_cache = true;
            if (preallocate) {
                ReservationFile reservation(dir);
                cfg.file_name = std::filesystem::path(reservation.path).filename().string();
                cfg.durable = true;
                cfg.preallocate = true;
                auto tier = std::make_unique<NvmeKvTier>(cfg);
                struct stat state {};
                if (::fstat(reservation.fd, &state) != 0 || state.st_size < 0 || state.st_blocks < 0 ||
                        uint64_t(state.st_size) < total || uint64_t(state.st_blocks) * 512 < total)
                    throw std::runtime_error("Filesystem did not physically reserve the requested NVMe quota");
                return tier;
            }
            return std::make_unique<NvmeKvTier>(cfg);
        }), py::arg("directory"), py::arg("total_bytes"), py::arg("slot_bytes"), py::arg("preallocate") = false)
        .def("allocate", [](NvmeKvTier &tier, uint32_t block) {
            tier.place_block(block);
            return tier.allocation(block);
        })
        .def("release", &NvmeKvTier::try_release_allocation)
        .def_property_readonly("used_slots", &NvmeKvTier::used_slots)
        .def("io", [](NvmeKvTier &tier, const std::vector<NvmeIoSpan> &spans,
                       py::buffer buffer, bool write) {
            auto info = buffer.request(!write);
            if (info.ndim != 1 || info.itemsize != 1 || info.strides[0] != 1)
                throw std::invalid_argument("NVMe IO requires contiguous one-dimensional bytes");
            NvmeBatchIoStats stats;
            {
                py::gil_scoped_release release;
                if (write) tier.write_spans(spans, info.ptr, info.size, &stats);
                else tier.read_spans(spans, info.ptr, info.size, &stats);
            }
            py::dict result;
            result["bytes"] = stats.bytes; result["syscalls"] = stats.syscalls;
            result["duration_ns"] = stats.duration_ns;
            return result;
        });
}
