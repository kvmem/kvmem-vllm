// Host-only bindings. Original algorithms remain in third_party/qw3.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "qw3/kvmem_store.hpp"
#include "qw3/nvme_kv_tier.hpp"
#include "qw3/kvmem_repository_gc.hpp"
#include "historical_payload.hpp"
#include "historical_manager.hpp"
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
#include <filesystem>
#include <cstring>
#include <limits>
#include <array>
#include <cmath>
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

PYBIND11_MODULE(_native, m) {
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
        size_t vectorized_records = 0;
        {
          py::gil_scoped_release release;
          for (const auto &r : rows) {
            const auto &a=r.raw, &b=r.sums;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
            if (vectorized && r.fp8 && mean_fp8_simd_eligible(a, b, r.count)) {
                finalize_mean_fp8_simd(a, b, r.count);
                ++vectorized_records;
                continue;
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
          }
        }
        if (vectorized) return py::int_(vectorized_records);
        return py::none();
    }, py::arg("jobs"), py::arg("vectorized") = false);
    m.def("scatter_history_batch", [](const std::vector<py::tuple> &jobs) {
        struct Row {
            py::buffer_info raw, values, dst_raw, dst_values, delta, sum;
            ssize_t begin, offset, count;
            bool copy_only;
        };
        std::vector<Row> rows;
        rows.reserve(jobs.size());
        for (const auto &job : jobs) {
            if (job.size() != 9 && job.size() != 10)
                throw std::invalid_argument("Invalid history scatter descriptor");
            if (job.size() == 10 && !PyBool_Check(job[9].ptr()))
                throw std::invalid_argument("History copy-only flag must be a bool");
            const bool copy_only = job.size() == 10 && job[9].cast<bool>();
            if (copy_only && !job[4].is_none())
                throw std::invalid_argument("History copy-only descriptor cannot contain a reduction");
            Row r{job[0].cast<py::buffer>().request(), job[1].cast<py::buffer>().request(),
                  job[2].cast<py::buffer>().request(), job[3].cast<py::buffer>().request(),
                  job[4].is_none() ? py::buffer_info{} : job[4].cast<py::buffer>().request(),
                  job[5].cast<py::buffer>().request(),
                  job[6].cast<ssize_t>(), job[7].cast<ssize_t>(), job[8].cast<ssize_t>(),
                  copy_only};
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
        // Every descriptor is validated before touching any destination. The
        // exported buffers retain the packet and each authority through this call.
        py::gil_scoped_release no_gil;
        for (const auto &r : rows) {
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
            }
            // The caller finalizes this completed page from its immutable raw
            // bytes before committing history. None without this explicit flag
            // retains the original small-decode fused-reduction semantics.
            if (r.copy_only) continue;
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
        }
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
        {
            py::gil_scoped_release release;
            for (size_t i = begin; i < end; ++i) {
                if (const char *source = pointers[i - begin]) {
                    std::memcpy(static_cast<char *>(target.ptr) + i * length, source, length);
                    copied += length;
                }
            }
        }
        return copied;
    });
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
        .value("RETRY", PayloadArena::RetireResult::Retry);
    py::class_<PayloadArena, std::shared_ptr<PayloadArena>>(m, "HistoricalPayloadArena")
        .def(py::init([](uint32_t capacity, size_t record_bytes, bool background,
                        uint32_t poll_ms, uint32_t scan_slots, uint32_t release_batch) {
            if (record_bytes > size_t(PY_SSIZE_T_MAX))
                throw std::invalid_argument("Historical CPU record exceeds buffer protocol limits");
            return std::make_shared<PayloadArena>(PayloadArena::Config{
                capacity, record_bytes, poll_ms, scan_slots, release_batch, background});
        }), py::arg("capacity"), py::arg("record_bytes"), py::arg("background") = true,
            py::arg("poll_ms") = 10, py::arg("scan_slots") = 256, py::arg("release_batch") = 64)
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
        .def("reclaimed", &PayloadArena::reclaimed)
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
    py::class_<HistoryWorker::Config>(m, "HistoricalGcConfig")
        .def(py::init<>())
#define HCFG(name) .def_readwrite(#name, &HistoryWorker::Config::name)
        HCFG(max_scopes) HCFG(scope_capacity) HCFG(poll_ms) HCFG(ready_low) HCFG(ready_high)
        HCFG(cpu_free_low) HCFG(cpu_free_high) HCFG(scan_blocks) HCFG(delete_blocks)
        HCFG(mode) HCFG(policy) HCFG(keep_mandatory) HCFG(background)
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
        .def("admit_read", [](HistoryScope &scope, KvMemBlockHandle handle) -> py::object {
            auto admission = scope.admit(handle);
            if (!admission.lease) return py::none();
            auto cpu = std::dynamic_pointer_cast<vllm_kvmem::HistoricalReadOwner>(admission.cpu);
            return py::make_tuple(std::move(admission.lease), cpu ? py::cast(cpu->read.clone()) : py::none());
        })
        .def("publish", &HistoryScope::publish, py::arg("lease"), py::arg("cpu"),
             py::arg("flags"), py::arg("use_seq") = 0)
        .def("update", &HistoryScope::update, py::arg("lease"), py::arg("flags"), py::arg("use_seq") = 0)
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
        .def("prepare_append", &KvMemStore::prepare_repository_append, py::keep_alive<0, 1>())
        .def("checkpoint", &KvMemStore::repository_checkpoint,
             py::arg("position"), py::arg("pin_working_set") = true)
        .def("checkpoint_valid", &KvMemStore::repository_checkpoint_valid)
        .def("working_records", [](const KvMemStore &store) {
            // Value tuples only: no block/Store views survive into a checkpoint.
            std::vector<std::tuple<uint32_t, uint64_t, uint64_t, uint32_t, int64_t>> records;
            for (const auto &block : store.blocks()) {
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
        .def("prepare_rewind", &KvMemStore::prepare_repository_rewind, py::keep_alive<0, 1>())
        .def("read", [](const KvMemStore &s, uint64_t b, uint32_t n) {
            return OwnedRead(s.repository_storage_read(b, n));
        })
        .def("handle", &KvMemStore::repository_handle)
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
        .def("select", &KvMemStore::set_repository_selection,
             py::arg("handles"), py::arg("force_raw_refresh") = false)
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
