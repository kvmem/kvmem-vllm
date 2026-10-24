#pragma once

// Private CPU FPS workers. No Torch scheduler, OpenMP state or Python callback
// is touched by a worker, and original FP8 bytes remain owned by the caller.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__FAST_MATH__)
#error "CPU FPS requires IEEE FP32 arithmetic; fast-math is unsupported"
#endif

namespace kvmem_fps_cpu {
namespace py = pybind11;

class BusyError : public std::runtime_error {
public:
    BusyError() : std::runtime_error("CPU FPS submission limit reached") {}
};

struct Geometry {
    size_t layers, blocks, tokens, heads, dim;
    std::array<size_t, 5> stride;
    size_t groups() const { return layers*blocks*heads; }
};

inline const std::array<float, 256> &fp8_table() {
    static const auto table = [] {
        std::array<float, 256> values{};
        for (unsigned code = 0; code < 256; ++code) {
            const unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
            float value = (exponent == 15 && mantissa == 7)
                ? std::numeric_limits<float>::quiet_NaN()
                : exponent == 0 ? std::ldexp(float(mantissa), -9)
                : std::ldexp(1.f + float(mantissa)*.125f, int(exponent)-7);
            values[code] = (code & 128) ? -value : value;
        }
        return values;
    }();
    return table;
}

// Independent lanes permit SIMD without reassociating an individual FP32
// addition. Explicitly forbid contraction: subtraction, square and addition
// must remain separate operations for the conservative reduction guard.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-fast-math", "fp-contract=off")))
#endif
inline float squared(const float *left, const float *right, size_t dim) {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif
    std::array<float, 8> sums{};
    size_t d = 0;
    for (; d+8 <= dim; d += 8) {
        for (size_t lane = 0; lane < 8; ++lane) {
            const float value = right ? left[d+lane]-right[d+lane] : left[d+lane];
            const float square = value*value;
            sums[lane] = sums[lane]+square;
        }
    }
    float result = 0.f;
    for (float value : sums) result = result+value;
    for (; d < dim; ++d) {
        const float value = right ? left[d]-right[d] : left[d];
        const float square = value*value;
        result = result+square;
    }
    return result;
}

struct Job {
    const uint8_t *raw;
    Geometry shape;
    std::vector<int64_t> counts;
    int8_t *offsets;
    bool *ambiguous;
    size_t next = 0, running = 0;
    bool done = false;
    std::exception_ptr error;
    std::condition_variable completed;
};

class Pool {
public:
    static constexpr size_t max_dim = 4096, max_tokens = 32;
    static constexpr size_t scratch_per_worker = max_dim*max_tokens*sizeof(float);

    Pool(size_t threads, size_t max_submissions) : workers_(threads), maximum_(max_submissions) {
        if (!threads || threads > 32 || !max_submissions || max_submissions > 2)
            throw std::invalid_argument("CPU FPS requires 1..32 workers and 1..2 submissions");
        // Allocate bounded worker scratch before starting any thread so an
        // allocation failure never escapes a worker or leaves orphan threads.
        scratch_.resize(threads);
        for (auto &scratch : scratch_) scratch.resize(max_dim*max_tokens);
        queue_.reserve(max_submissions);
        (void)fp8_table();
        try {
            for (size_t worker = 0; worker < workers_; ++worker)
                threads_.emplace_back([this, worker] { work(worker); });
        } catch (...) {
            close();
            throw;
        }
    }

    Pool(const Pool &) = delete;
    Pool &operator=(const Pool &) = delete;
    ~Pool() { close(); }

    void select(const uint8_t *raw, Geometry shape, std::vector<int64_t> counts,
                int8_t *offsets, bool *ambiguous) {
        auto job = std::make_shared<Job>();
        job->raw = raw; job->shape = shape; job->counts = std::move(counts);
        job->offsets = offsets; job->ambiguous = ambiguous;
        std::unique_lock<std::mutex> lock(mutex_);
        if (closing_) throw std::runtime_error("CPU FPS pool is closed");
        if (active_ >= maximum_) {
            ++busy_;
            throw BusyError();
        }
        queue_.push_back(job);
        ++active_; ++submitted_; peak_ = std::max(peak_, active_);
        wake_.notify_all();
        job->completed.wait(lock, [&] { return job->done; });
        // Every worker using this job has completed before input/output Python
        // owners can be released, including after any worker's first error.
        if (job->error) std::rethrow_exception(job->error);
    }

    void close() {
        std::lock_guard<std::mutex> closing_lock(close_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closing_ = true;
        }
        wake_.notify_all();
        for (auto &thread : threads_)
            if (thread.joinable()) thread.join();
    }

    py::dict stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        py::dict result;
        result["workers"] = workers_; result["max_submissions"] = maximum_;
        result["active"] = active_; result["peak"] = peak_;
        result["jobs"] = submitted_; result["completed_jobs"] = completed_;
        result["failed_jobs"] = failed_; result["groups"] = groups_;
        result["busy_rejections"] = busy_; result["closed"] = closing_;
        result["scratch_bytes"] = workers_*scratch_per_worker;
        result["scratch_bytes_per_worker"] = scratch_per_worker;
        return result;
    }

private:
    static void group(Job &job, size_t group_index, std::vector<float> &keys) {
        const auto &s = job.shape;
        const size_t head = group_index%s.heads;
        const size_t block = (group_index/s.heads)%s.blocks;
        const size_t layer = group_index/(s.heads*s.blocks);
        const size_t count = size_t(job.counts[block]);
        for (size_t slot = 0; slot < 4; ++slot)
            job.offsets[((layer*s.blocks+block)*4+slot)*s.heads+head] = -1;
        job.ambiguous[group_index] = false;
        if (!count) return;
        const auto &table = fp8_table();
        const uint8_t *base = job.raw+layer*s.stride[0]+block*s.stride[1]+head*s.stride[3];
        for (size_t token = 0; token < count; ++token) {
            const uint8_t *source = base+token*s.stride[2];
            for (size_t d = 0; d < s.dim; ++d) {
                const uint8_t code = source[d*s.stride[4]];
                if ((code & 127) == 127)
                    throw std::invalid_argument("Nonfinite original FP8 Key in CPU FPS");
                keys[token*s.dim+d] = table[code];
            }
        }
        std::array<float, max_tokens> criterion{}, nearest{};
        std::array<bool, max_tokens> available{};
        for (size_t token = 0; token < count; ++token) {
            criterion[token] = squared(keys.data()+token*s.dim, nullptr, s.dim);
            available[token] = true;
        }
        for (size_t slot = 0; slot < std::min<size_t>(4, count); ++slot) {
            float best = -std::numeric_limits<float>::infinity();
            float runner_up = best;
            size_t chosen = 0;
            for (size_t token = 0; token < count; ++token) {
                if (!available[token]) continue;
                const float candidate = criterion[token];
                if (!std::isfinite(candidate))
                    throw std::invalid_argument("Nonfinite FP32 CPU FPS criterion");
                if (candidate > best) {
                    runner_up = best; best = candidate; chosen = token;
                } else if (candidate > runner_up) {
                    runner_up = candidate;
                }
            }
            const float tolerance = 4.f*float(s.dim)*std::numeric_limits<float>::epsilon()*std::fabs(best);
            if (std::isfinite(runner_up) && best-runner_up <= tolerance)
                job.ambiguous[group_index] = true;
            job.offsets[((layer*s.blocks+block)*4+slot)*s.heads+head] = int8_t(chosen);
            available[chosen] = false;
            if (slot == 3) break;
            for (size_t token = 0; token < count; ++token) {
                const float distance = squared(keys.data()+token*s.dim,
                    keys.data()+chosen*s.dim, s.dim);
                if (!std::isfinite(distance))
                    throw std::invalid_argument("Nonfinite FP32 CPU FPS distance");
                nearest[token] = slot == 0 ? distance : std::min(nearest[token], distance);
                criterion[token] = nearest[token];
            }
        }
    }

    void work(size_t worker) noexcept {
        for (;;) {
            std::shared_ptr<Job> job;
            size_t begin, end;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return closing_ || !queue_.empty(); });
                if (queue_.empty()) return;
                job = queue_.front();
                begin = job->next;
                end = std::min(begin+8, job->shape.groups());
                job->next = end;
                ++job->running;
                if (end < job->shape.groups())
                    std::rotate(queue_.begin(), queue_.begin()+1, queue_.end());
                else queue_.erase(queue_.begin());
            }
            std::exception_ptr failure;
            size_t finished = 0;
            try {
                for (size_t g = begin; g < end; ++g) {
                    group(*job, g, scratch_[worker]);
                    ++finished;
                }
            } catch (...) { failure = std::current_exception(); }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                groups_ += finished;
                if (failure && !job->error) {
                    job->error = failure;
                    job->next = job->shape.groups();
                    queue_.erase(std::remove(queue_.begin(), queue_.end(), job), queue_.end());
                }
                --job->running;
                if (job->next == job->shape.groups() && !job->running) {
                    job->done = true;
                    --active_;
                    if (job->error) ++failed_; else ++completed_;
                    job->completed.notify_one();
                }
            }
        }
    }

    const size_t workers_, maximum_;
    mutable std::mutex mutex_;
    std::mutex close_mutex_;
    std::condition_variable wake_;
    std::vector<std::shared_ptr<Job>> queue_;
    std::vector<std::thread> threads_;
    std::vector<std::vector<float>> scratch_;
    size_t active_ = 0, peak_ = 0, submitted_ = 0, completed_ = 0, failed_ = 0, groups_ = 0, busy_ = 0;
    bool closing_ = false;
};

inline Geometry geometry(const py::buffer_info &info, const std::vector<int64_t> &counts) {
    if (info.ndim != 5 || info.itemsize != 1 || info.format != py::format_descriptor<uint8_t>::format())
        throw std::invalid_argument("CPU FPS requires a five-dimensional uint8 FP8 buffer");
    size_t elements = 1;
    std::vector<std::pair<size_t, size_t>> axes;
    for (size_t axis = 0; axis < 5; ++axis) {
        if (info.shape[axis] <= 0 || info.strides[axis] < 0)
            throw std::invalid_argument("CPU FPS requires positive dimensions and dense positive strides");
        const size_t extent = size_t(info.shape[axis]);
        if (extent > (64u*1024u*1024u)/elements)
            throw std::invalid_argument("CPU FPS input exceeds 64 MiB");
        elements *= extent;
        if (extent > 1) axes.emplace_back(size_t(info.strides[axis]), extent);
    }
    // Dense axis permutations cover both [L,B,T,H,D] and the archive's B/L
    // transpose. Reject holes/overlap/negative strides before pointer access.
    std::sort(axes.begin(), axes.end());
    size_t stride = 1;
    for (auto axis : axes) {
        if (axis.first != stride)
            throw std::invalid_argument("CPU FPS requires dense nonoverlapping byte strides");
        stride *= axis.second;
    }
    Geometry shape{size_t(info.shape[0]), size_t(info.shape[1]), size_t(info.shape[2]),
        size_t(info.shape[3]), size_t(info.shape[4]), {}};
    if (shape.blocks > 32 || shape.tokens > 32 || shape.dim > 4096 || counts.size() != shape.blocks)
        throw std::invalid_argument("Invalid CPU FPS block/token/dimension/count geometry");
    for (size_t axis = 0; axis < 5; ++axis) shape.stride[axis] = size_t(info.strides[axis]);
    for (int64_t count : counts)
        if (count < 0 || size_t(count) > shape.tokens)
            throw std::invalid_argument("CPU FPS valid count exceeds original token extent");
    return shape;
}

inline void bind(py::module_ &m) {
    py::register_exception<BusyError>(m, "KeyFPSCPUBusyError", PyExc_RuntimeError);
    py::class_<Pool>(m, "KeyFPSCPUPool")
        .def(py::init<size_t, size_t>(), py::arg("threads") = 8, py::arg("max_submissions") = 2)
        .def("select", [](Pool &pool, py::buffer raw, const std::vector<int64_t> &counts) {
            auto info = raw.request();
            auto shape = geometry(info, counts);
            py::array_t<int8_t> offsets({py::ssize_t(shape.layers), py::ssize_t(shape.blocks),
                py::ssize_t(4), py::ssize_t(shape.heads)});
            py::array_t<bool> ambiguous({py::ssize_t(shape.layers), py::ssize_t(shape.blocks),
                py::ssize_t(shape.heads)});
            auto *out = offsets.mutable_data();
            auto *guard = ambiguous.mutable_data();
            {
                py::gil_scoped_release release;
                pool.select(static_cast<const uint8_t *>(info.ptr), shape, counts,
                    out, guard);
            }
            return py::make_tuple(std::move(offsets), std::move(ambiguous));
        }, py::arg("raw"), py::arg("counts"))
        .def("stats", &Pool::stats)
        .def("close", [](Pool &pool) { py::gil_scoped_release release; pool.close(); });
}

} // namespace kvmem_fps_cpu
