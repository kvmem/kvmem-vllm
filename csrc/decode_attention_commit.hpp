// Accepted output rows stay in their downloaded dense matrix. Validation and
// staging never mutate the published dictionary or advance the scope parser.
#pragma once
#include "block_graph_accept.hpp"
#include "parallel_rows.hpp"

namespace kvmem_decode {
namespace py = pybind11;

inline py::tuple prepare(py::dict old, size_t samples, const std::vector<size_t>& indices,
        const std::vector<int64_t>& targets, py::buffer masses, size_t max_bytes) {
    const auto b = masses.request();
    const bool fp32 = b.format == py::format_descriptor<float>::format() && b.itemsize == 4;
    const bool fp64 = b.format == py::format_descriptor<double>::format() && b.itemsize == 8;
    if (b.ndim != 2 || (!fp32 && !fp64) || b.shape[1] != static_cast<ssize_t>(targets.size()))
        throw std::invalid_argument("Decode attention matrix shape/dtype mismatch");
    if (old.size() > max_bytes/320 || targets.size() > max_bytes/320 ||
            4096 + (old.size()+targets.size())*320 > max_bytes)
        throw std::length_error("Decode attention staging capacity exceeded before acceptance");
    std::unordered_map<int64_t,double> updated;
    for (auto item : old) updated.emplace(py::cast<int64_t>(item.first), py::cast<double>(item.second));
    std::vector<int64_t> additions;
    const auto read = fp32 ? &kvmem_graph::read<float> : &kvmem_graph::read<double>;
    {
        py::gil_scoped_release release;
        size_t previous = 0; bool first = true;
        for (auto row : indices) {
            if (row >= static_cast<size_t>(b.shape[0]) || (!first && row <= previous))
                throw std::invalid_argument("Invalid or duplicate decode accepted row index");
            first = false; previous = row;
            double sum = 0., correction = 0.;
            for (size_t col=0; col<targets.size(); ++col) {
                const double value = read(b,row,col);
                if (!std::isfinite(value) || value < 0.)
                    throw std::invalid_argument("Action attention mass is invalid or does not sum to one");
                const double next = sum+value;
#if PY_VERSION_HEX >= 0x030C0000
                correction += std::abs(sum) >= std::abs(value) ? (sum-next)+value : (value-next)+sum;
#endif
                sum = next;
            }
            sum += correction;
            if (!std::isfinite(sum) || std::abs(sum-1.) > .005)
                throw std::invalid_argument("Action attention mass is invalid or does not sum to one");
            for (size_t col=0; col<targets.size(); ++col) {
                const double value = read(b,row,col);
                // Preserve Python's sequential double additions, including
                // repeated fragments belonging to the same historical page.
                if (value != 0.) {
                    auto item = updated.emplace(targets[col],0.);
                    if (item.second) additions.push_back(targets[col]);
                    item.first->second += value;
                }
            }
        }
    }
    py::dict result = old.attr("copy")();
    for (auto key : additions) result[py::int_(key)] = py::float_(updated.at(key));
    for (const auto& item : updated) result[py::int_(item.first)] = py::float_(item.second);
    return py::make_tuple(result,samples+indices.size());
}

// KVMEM_NATIVE_DECODE_SPARSE_MASS: prepare() without the final rewrite of every
// unchanged entry. Same validation, same sequential additions, same result
// contents and insertion order: only keys whose value changed, new keys and old
// non-float values (prepare() converts those to float) are written after copy().
inline py::tuple prepare_sparse(py::dict old, size_t samples, const std::vector<size_t>& indices,
        const std::vector<int64_t>& targets, py::buffer masses, size_t max_bytes) {
    const auto b = masses.request();
    const bool fp32 = b.format == py::format_descriptor<float>::format() && b.itemsize == 4;
    const bool fp64 = b.format == py::format_descriptor<double>::format() && b.itemsize == 8;
    if (b.ndim != 2 || (!fp32 && !fp64) || b.shape[1] != static_cast<ssize_t>(targets.size()))
        throw std::invalid_argument("Decode attention matrix shape/dtype mismatch");
    if (old.size() > max_bytes/320 || targets.size() > max_bytes/320 ||
            4096 + (old.size()+targets.size())*320 > max_bytes)
        throw std::length_error("Decode attention staging capacity exceeded before acceptance");
    // value, state: 0 unchanged old entry, 1 changed old entry, 2 new key.
    std::unordered_map<int64_t,std::pair<double,char>> updated;
    std::vector<int64_t> rewrite;
    updated.reserve(old.size() + targets.size());
    for (auto item : old) {
        const int64_t key = py::cast<int64_t>(item.first);
        updated.emplace(key, std::make_pair(py::cast<double>(item.second), char{0}));
        if (!PyFloat_CheckExact(item.second.ptr())) rewrite.push_back(key);
    }
    std::vector<int64_t> additions, touched;
    const auto read = fp32 ? &kvmem_graph::read<float> : &kvmem_graph::read<double>;
    {
        py::gil_scoped_release release;
        size_t previous = 0; bool first = true;
        for (auto row : indices) {
            if (row >= static_cast<size_t>(b.shape[0]) || (!first && row <= previous))
                throw std::invalid_argument("Invalid or duplicate decode accepted row index");
            first = false; previous = row;
            double sum = 0., correction = 0.;
            for (size_t col=0; col<targets.size(); ++col) {
                const double value = read(b,row,col);
                if (!std::isfinite(value) || value < 0.)
                    throw std::invalid_argument("Action attention mass is invalid or does not sum to one");
                const double next = sum+value;
#if PY_VERSION_HEX >= 0x030C0000
                correction += std::abs(sum) >= std::abs(value) ? (sum-next)+value : (value-next)+sum;
#endif
                sum = next;
            }
            sum += correction;
            if (!std::isfinite(sum) || std::abs(sum-1.) > .005)
                throw std::invalid_argument("Action attention mass is invalid or does not sum to one");
            for (size_t col=0; col<targets.size(); ++col) {
                const double value = read(b,row,col);
                if (value != 0.) {
                    auto item = updated.emplace(targets[col], std::make_pair(0., char{2}));
                    if (item.second) additions.push_back(targets[col]);
                    else if (item.first->second.second == 0) {
                        item.first->second.second = 1;
                        touched.push_back(targets[col]);
                    }
                    item.first->second.first += value;
                }
            }
        }
    }
    py::dict result = old.attr("copy")();
    for (auto key : additions) result[py::int_(key)] = py::float_(updated.at(key).first);
    for (auto key : touched) result[py::int_(key)] = py::float_(updated.at(key).first);
    for (auto key : rewrite) result[py::int_(key)] = py::float_(updated.at(key).first);
    return py::make_tuple(result,samples+indices.size());
}

// KVMEM_NATIVE_DECODE_MASS_MAP: the accepted-mass dictionary kept natively.
// keys/values are in Python dict insertion order; index is (key, position)
// sorted by key. Copying is three flat vectors, so staging a prepare no longer
// rebuilds a Python dict of every attended block.
struct MassMap {
    std::vector<int64_t> keys;
    std::vector<double> values;
    std::vector<std::pair<int64_t,uint32_t>> index;

    size_t size() const { return keys.size(); }
    int64_t find(int64_t key) const {
        auto it = std::lower_bound(index.begin(), index.end(), std::make_pair(key, uint32_t{0}),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        return it != index.end() && it->first == key ? static_cast<int64_t>(it->second) : -1;
    }
    static MassMap from_dict(const py::dict& old) {
        MassMap out;
        out.keys.reserve(old.size()); out.values.reserve(old.size()); out.index.reserve(old.size());
        for (auto item : old) {
            // The same casts, in the same order, as prepare().
            const int64_t key = py::cast<int64_t>(item.first);
            const double value = py::cast<double>(item.second);
            out.index.emplace_back(key, static_cast<uint32_t>(out.keys.size()));
            out.keys.push_back(key);
            out.values.push_back(value);
        }
        std::sort(out.index.begin(), out.index.end());
        return out;
    }
    py::dict to_dict() const {
        py::dict out;
        for (size_t i = 0; i < keys.size(); ++i) {
            py::object k = py::reinterpret_steal<py::object>(PyLong_FromLongLong(keys[i]));
            py::object v = py::reinterpret_steal<py::object>(PyFloat_FromDouble(values[i]));
            if (!k || !v || PyDict_SetItem(out.ptr(), k.ptr(), v.ptr()) < 0) throw py::error_already_set();
        }
        return out;
    }
    // KVMEM_SESSION_FAST_TURN: tuple(sorted(to_dict().items())) without the
    // dictionary (keys are unique, index is already sorted by key).
    py::tuple sorted_rows() const {
        py::tuple out(index.size());
        for (size_t i = 0; i < index.size(); ++i) {
            PyObject* k = PyLong_FromLongLong(index[i].first);
            PyObject* v = PyFloat_FromDouble(values[index[i].second]);
            PyObject* row = k && v ? PyTuple_Pack(2, k, v) : nullptr;
            Py_XDECREF(k); Py_XDECREF(v);
            if (!row) throw py::error_already_set();
            PyTuple_SET_ITEM(out.ptr(), static_cast<Py_ssize_t>(i), row);
        }
        return out;
    }
    // KVMEM_SESSION_FAST_TURN: from_dict(dict(rows)) for checkpoint rows, i.e.
    // an exact tuple of exact (int, float) tuples with strictly increasing
    // int64 keys. None for anything else; the caller keeps the dict path.
    static py::object from_rows(const py::object& rows) {
        if (!PyTuple_CheckExact(rows.ptr())) return py::none();
        const Py_ssize_t count = PyTuple_GET_SIZE(rows.ptr());
        if (static_cast<uint64_t>(count) > UINT32_MAX) return py::none();
        MassMap out;
        out.keys.reserve(count); out.values.reserve(count); out.index.reserve(count);
        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject* row = PyTuple_GET_ITEM(rows.ptr(), i);
            if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2) return py::none();
            PyObject* k = PyTuple_GET_ITEM(row, 0);
            PyObject* v = PyTuple_GET_ITEM(row, 1);
            if (!PyLong_CheckExact(k) || !PyFloat_CheckExact(v)) return py::none();
            int overflow = 0;
            const long long key = PyLong_AsLongLongAndOverflow(k, &overflow);
            if (overflow) return py::none();
            if (i && key <= out.keys.back()) return py::none();
            out.index.emplace_back(key, static_cast<uint32_t>(i));
            out.keys.push_back(key);
            out.values.push_back(PyFloat_AS_DOUBLE(v));
        }
        return py::cast(std::move(out));
    }
};

// prepare() over a MassMap (or a dict, converted at the point prepare() casts
// it). Same checks in the same order, the same sequential double additions,
// and the same resulting contents and insertion order as prepare()'s dict.
inline py::tuple prepare_map(py::object old, size_t samples, const std::vector<size_t>& indices,
        const std::vector<int64_t>& targets, py::buffer masses, size_t max_bytes) {
    const auto b = masses.request();
    const bool fp32 = b.format == py::format_descriptor<float>::format() && b.itemsize == 4;
    const bool fp64 = b.format == py::format_descriptor<double>::format() && b.itemsize == 8;
    if (b.ndim != 2 || (!fp32 && !fp64) || b.shape[1] != static_cast<ssize_t>(targets.size()))
        throw std::invalid_argument("Decode attention matrix shape/dtype mismatch");
    const bool native = py::isinstance<MassMap>(old);
    if (!native && !py::isinstance<py::dict>(old))
        throw py::type_error("Decode attention mass must be a dict or DecodeMassMap");
    const size_t old_size = native ? old.cast<const MassMap&>().size() : py::len(old);
    if (old_size > max_bytes/320 || targets.size() > max_bytes/320 ||
            4096 + (old_size+targets.size())*320 > max_bytes)
        throw std::length_error("Decode attention staging capacity exceeded before acceptance");
    MassMap result = native ? old.cast<const MassMap&>() : MassMap::from_dict(old.cast<py::dict>());
    std::vector<std::pair<int64_t,uint32_t>> added;
    std::unordered_map<int64_t,uint32_t> added_at;
    const auto read = fp32 ? &kvmem_graph::read<float> : &kvmem_graph::read<double>;
    {
        py::gil_scoped_release release;
        size_t previous = 0; bool first = true;
        for (auto row : indices) {
            if (row >= static_cast<size_t>(b.shape[0]) || (!first && row <= previous))
                throw std::invalid_argument("Invalid or duplicate decode accepted row index");
            first = false; previous = row;
            double sum = 0., correction = 0.;
            for (size_t col=0; col<targets.size(); ++col) {
                const double value = read(b,row,col);
                if (!std::isfinite(value) || value < 0.)
                    throw std::invalid_argument("Action attention mass is invalid or does not sum to one");
                const double next = sum+value;
#if PY_VERSION_HEX >= 0x030C0000
                correction += std::abs(sum) >= std::abs(value) ? (sum-next)+value : (value-next)+sum;
#endif
                sum = next;
            }
            sum += correction;
            if (!std::isfinite(sum) || std::abs(sum-1.) > .005)
                throw std::invalid_argument("Action attention mass is invalid or does not sum to one");
            for (size_t col=0; col<targets.size(); ++col) {
                const double value = read(b,row,col);
                if (value == 0.) continue;
                const int64_t key = targets[col];
                int64_t pos = result.find(key);
                if (pos < 0) {
                    auto hit = added_at.find(key);
                    if (hit != added_at.end()) pos = hit->second;
                }
                if (pos < 0) {
                    // prepare(): emplace(key, 0.) then += value; first touch order.
                    pos = static_cast<int64_t>(result.keys.size());
                    result.keys.push_back(key);
                    result.values.push_back(0.);
                    added_at.emplace(key, static_cast<uint32_t>(pos));
                    added.emplace_back(key, static_cast<uint32_t>(pos));
                }
                result.values[static_cast<size_t>(pos)] += value;
            }
        }
        if (!added.empty()) {
            std::sort(added.begin(), added.end());
            std::vector<std::pair<int64_t,uint32_t>> merged;
            merged.reserve(result.index.size() + added.size());
            std::merge(result.index.begin(), result.index.end(), added.begin(), added.end(),
                std::back_inserter(merged));
            result.index.swap(merged);
        }
    }
    return py::make_tuple(py::cast(std::move(result)), samples+indices.size());
}

// KVMEM_NATIVE_DECODE_LAYER_BATCH: prepare_map() with the targets read from a
// 1-D int64 array (one row of decode_identities_from_maps) instead of a Python
// list. The same checks in the same order, the same sequential double
// additions per key and the same resulting contents and insertion order.
// Each column's place in the map is looked up once per call: by one merge walk
// over the key-sorted index when the targets are non-decreasing, otherwise by
// the same binary search as prepare_map().
template<class T>
inline double read_mass(const py::buffer_info& b, size_t row, size_t col) {
    T value;
    std::memcpy(&value, static_cast<const char*>(b.ptr) + static_cast<ssize_t>(row)*b.strides[0] +
                static_cast<ssize_t>(col)*b.strides[1], sizeof(T));
    return static_cast<double>(value);
}

template<class T>
inline void accumulate_rows(const py::buffer_info& b, const std::vector<size_t>& indices,
        const std::vector<int64_t>& targets, MassMap& result) {
    const size_t columns = targets.size();
    // slot >= 0: position of the key in the staged map; slot < 0: -(1 + id) of
    // a key the old map does not hold (its position is assigned on first touch).
    std::vector<int64_t> slot(columns);
    size_t fresh = 0;
    bool ordered = true;
    for (size_t col = 1; col < columns && ordered; ++col) ordered = targets[col-1] <= targets[col];
    if (ordered) {
        const auto& index = result.index;
        size_t at = 0;
        for (size_t col = 0; col < columns; ++col) {
            const int64_t key = targets[col];
            if (col && key == targets[col-1]) { slot[col] = slot[col-1]; continue; }
            while (at < index.size() && index[at].first < key) ++at;
            slot[col] = at < index.size() && index[at].first == key
                ? static_cast<int64_t>(index[at].second) : -static_cast<int64_t>(++fresh);
        }
    } else {
        std::unordered_map<int64_t,int64_t> ids;
        for (size_t col = 0; col < columns; ++col) {
            const int64_t pos = result.find(targets[col]);
            if (pos >= 0) { slot[col] = pos; continue; }
            auto item = ids.emplace(targets[col], static_cast<int64_t>(fresh + 1));
            if (item.second) ++fresh;
            slot[col] = -item.first->second;
        }
    }
    std::vector<int64_t> fresh_at(fresh, -1);
    std::vector<std::pair<int64_t,uint32_t>> added;
    size_t previous = 0; bool first = true;
    for (auto row : indices) {
        if (row >= static_cast<size_t>(b.shape[0]) || (!first && row <= previous))
            throw std::invalid_argument("Invalid or duplicate decode accepted row index");
        first = false; previous = row;
        double sum = 0., correction = 0.;
        for (size_t col = 0; col < columns; ++col) {
            const double value = read_mass<T>(b, row, col);
            if (!std::isfinite(value) || value < 0.)
                throw std::invalid_argument("Action attention mass is invalid or does not sum to one");
            const double next = sum+value;
#if PY_VERSION_HEX >= 0x030C0000
            correction += std::abs(sum) >= std::abs(value) ? (sum-next)+value : (value-next)+sum;
#endif
            sum = next;
        }
        sum += correction;
        if (!std::isfinite(sum) || std::abs(sum-1.) > .005)
            throw std::invalid_argument("Action attention mass is invalid or does not sum to one");
        for (size_t col = 0; col < columns; ++col) {
            const double value = read_mass<T>(b, row, col);
            if (value == 0.) continue;
            int64_t pos = slot[col];
            if (pos < 0) {
                int64_t& assigned = fresh_at[static_cast<size_t>(-pos - 1)];
                if (assigned < 0) {
                    // prepare(): emplace(key, 0.) then += value; first touch order.
                    assigned = static_cast<int64_t>(result.keys.size());
                    result.keys.push_back(targets[col]);
                    result.values.push_back(0.);
                    added.emplace_back(targets[col], static_cast<uint32_t>(assigned));
                }
                pos = assigned;
            }
            result.values[static_cast<size_t>(pos)] += value;
        }
    }
    if (!added.empty()) {
        std::sort(added.begin(), added.end());
        std::vector<std::pair<int64_t,uint32_t>> merged;
        merged.reserve(result.index.size() + added.size());
        std::merge(result.index.begin(), result.index.end(), added.begin(), added.end(),
            std::back_inserter(merged));
        result.index.swap(merged);
    }
}

inline py::tuple prepare_map_targets(py::object old, size_t samples, const std::vector<size_t>& indices,
        py::object targets_o, py::buffer masses, size_t max_bytes) {
    if (!py::isinstance<py::array_t<int64_t>>(targets_o))
        throw py::type_error("Decode attention targets must be a 1-D int64 array");
    const py::array targets_a = py::reinterpret_borrow<py::array>(targets_o);
    if (targets_a.ndim() != 1 || !(targets_a.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_))
        throw py::type_error("Decode attention targets must be a 1-D int64 array");
    std::vector<int64_t> targets(static_cast<size_t>(targets_a.shape(0)));
    {
        const char* base = static_cast<const char*>(targets_a.data());
        const ssize_t stride = targets_a.strides(0);
        for (size_t i = 0; i < targets.size(); ++i)
            std::memcpy(&targets[i], base + static_cast<ssize_t>(i)*stride, sizeof(int64_t));
    }
    const auto b = masses.request();
    const bool fp32 = b.format == py::format_descriptor<float>::format() && b.itemsize == 4;
    const bool fp64 = b.format == py::format_descriptor<double>::format() && b.itemsize == 8;
    if (b.ndim != 2 || (!fp32 && !fp64) || b.shape[1] != static_cast<ssize_t>(targets.size()))
        throw std::invalid_argument("Decode attention matrix shape/dtype mismatch");
    const bool native = py::isinstance<MassMap>(old);
    if (!native && !py::isinstance<py::dict>(old))
        throw py::type_error("Decode attention mass must be a dict or DecodeMassMap");
    const size_t old_size = native ? old.cast<const MassMap&>().size() : py::len(old);
    if (old_size > max_bytes/320 || targets.size() > max_bytes/320 ||
            4096 + (old_size+targets.size())*320 > max_bytes)
        throw std::length_error("Decode attention staging capacity exceeded before acceptance");
    MassMap result = native ? old.cast<const MassMap&>() : MassMap::from_dict(old.cast<py::dict>());
    {
        py::gil_scoped_release release;
        if (fp32) accumulate_rows<float>(b, indices, targets, result);
        else accumulate_rows<double>(b, indices, targets, result);
    }
    return py::make_tuple(py::cast(std::move(result)), samples+indices.size());
}

// KVMEM_NATIVE_DECODE_PREPARE_BATCH: prepare_map_targets() for every layer of
// one request in one call. olds holds each layer's DecodeMassMap, targets is the
// int64 [layers, columns] identity array of decode_identities_from_maps, masses
// one float32/float64 [layers, rows, columns] buffer, and every layer folds the
// same accepted rows. Each layer runs prepare_map_targets()'s capacity check,
// copy and accumulate_rows() unchanged, so every map is bitwise equal to the
// per-layer call; the layers are independent and run on the publication pool
// (KVMEM_NATIVE_PUBLISH_THREADS) with the GIL released. A failing layer raises
// what its per-layer call raises, the first one in layer order. Returns the
// staged maps, or None when a map is not a DecodeMassMap or the arrays do not
// have these shapes (the caller then runs the per-layer path).
inline py::object prepare_maps_targets(py::sequence olds, const std::vector<size_t>& indices,
        py::object targets_o, py::buffer masses, size_t max_bytes) {
    if (!py::isinstance<py::array_t<int64_t>>(targets_o)) return py::none();
    const py::array targets_a = py::reinterpret_borrow<py::array>(targets_o);
    if (targets_a.ndim() != 2 || !(targets_a.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_))
        return py::none();
    const size_t layers = static_cast<size_t>(targets_a.shape(0));
    const size_t columns = static_cast<size_t>(targets_a.shape(1));
    if (layers == 0 || static_cast<size_t>(py::len(olds)) != layers) return py::none();
    const auto b = masses.request();
    const bool fp32 = b.format == py::format_descriptor<float>::format() && b.itemsize == 4;
    const bool fp64 = b.format == py::format_descriptor<double>::format() && b.itemsize == 8;
    if (b.ndim != 3 || (!fp32 && !fp64) || b.shape[0] != static_cast<ssize_t>(layers) ||
            b.shape[2] != static_cast<ssize_t>(columns))
        return py::none();
    std::vector<const MassMap*> sources(layers);
    std::vector<std::vector<int64_t>> targets(layers, std::vector<int64_t>(columns));
    std::vector<py::buffer_info> views;
    views.reserve(layers);
    const char* target_base = static_cast<const char*>(targets_a.data());
    for (size_t layer = 0; layer < layers; ++layer) {
        py::handle old = olds[layer];
        if (!py::isinstance<MassMap>(old)) return py::none();
        // The sequence keeps each map alive; maps are never mutated once made.
        sources[layer] = &old.cast<const MassMap&>();
        for (size_t col = 0; col < columns; ++col)
            std::memcpy(&targets[layer][col], target_base + static_cast<ssize_t>(layer)*targets_a.strides(0) +
                        static_cast<ssize_t>(col)*targets_a.strides(1), sizeof(int64_t));
        // The [rows, columns] matrix prepare_map_targets() reads for this layer.
        views.emplace_back(static_cast<char*>(b.ptr) + static_cast<ssize_t>(layer)*b.strides[0], b.itemsize,
                           b.format, 2, std::vector<ssize_t>{b.shape[1], b.shape[2]},
                           std::vector<ssize_t>{b.strides[1], b.strides[2]}, true);
    }
    std::vector<MassMap> results(layers);
    {
        py::gil_scoped_release release;
        kvmem_parallel::for_rows(layers, [&](size_t layer) {
            const size_t old_size = sources[layer]->size();
            if (old_size > max_bytes/320 || columns > max_bytes/320 ||
                    4096 + (old_size+columns)*320 > max_bytes)
                throw std::length_error("Decode attention staging capacity exceeded before acceptance");
            MassMap result = *sources[layer];
            if (fp32) accumulate_rows<float>(views[layer], indices, targets[layer], result);
            else accumulate_rows<double>(views[layer], indices, targets[layer], result);
            results[layer] = std::move(result);
        });
    }
    py::list out(layers);
    for (size_t layer = 0; layer < layers; ++layer)
        out[layer] = py::cast(std::move(results[layer]));
    return std::move(out);
}

inline void bind(py::module_& module) {
    module.def("prepare_decode_attention",&prepare);
    module.def("prepare_decode_attention_sparse",&prepare_sparse);
    py::class_<MassMap>(module, "DecodeMassMap")
        .def_static("from_dict", &MassMap::from_dict)
        .def("to_dict", &MassMap::to_dict)
        .def("sorted_rows", &MassMap::sorted_rows)
        // KVMEM_CHECKPOINT_CAPTURE_INDEX: max(to_dict()) without the dict, or
        // None when empty (index is sorted by key).
        .def("max_key", [](const MassMap& m) -> py::object {
            if (m.index.empty()) return py::none();
            return py::int_(m.index.back().first);
        })
        // KVMEM_CHECKPOINT_CAPTURE_INDEX: from_rows(sorted_rows()) without the
        // Python rows, i.e. the same entries re-inserted in key order.
        .def("sorted_copy", [](const MassMap& m) {
            MassMap out;
            out.keys.reserve(m.index.size()); out.values.reserve(m.index.size());
            out.index.reserve(m.index.size());
            for (size_t i = 0; i < m.index.size(); ++i) {
                out.index.emplace_back(m.index[i].first, static_cast<uint32_t>(i));
                out.keys.push_back(m.index[i].first);
                out.values.push_back(m.values[m.index[i].second]);
            }
            return out;
        })
        .def_static("from_rows", &MassMap::from_rows)
        .def("__len__", &MassMap::size);
    module.def("prepare_decode_attention_map", &prepare_map);
    module.def("prepare_decode_attention_map_targets", &prepare_map_targets);
    module.def("prepare_decode_attention_maps_targets", &prepare_maps_targets);
}
}
