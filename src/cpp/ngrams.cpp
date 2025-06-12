#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <parallel_hashmap/phmap.h>

#include <string>
#include <string_view>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

namespace fastgrams {

class ArrowStringArrayMapper {
public:
    explicit ArrowStringArrayMapper(py::dict vocab) {
        for (auto kv : vocab) {
            vocab_[py::cast<std::string>(kv.first)] =
                static_cast<std::int64_t>(py::cast<std::int64_t>(kv.second));
        }
    }

    // Map Arrow StringArray buffers (utf8_data + offsets) to int64 IDs. Unknowns → -1.
    py::array_t<std::int64_t> map_ids(
        py::bytes utf8_data,
        py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> offsets) const {

        std::string data = utf8_data;
        const char* base = data.data();
        std::size_t data_size = data.size();

        auto off = offsets.unchecked<1>();
        if (off.shape(0) < 1)
            throw std::runtime_error("offsets must contain at least one element");

        std::size_t n_strings = static_cast<std::size_t>(off.shape(0) - 1);
        py::array_t<std::int64_t> out(n_strings);
        auto out_buf = out.mutable_unchecked<1>();

        for (std::size_t i = 0; i < n_strings; ++i) {
            std::int32_t start = off(i);
            std::int32_t end   = off(i + 1);

            if (start < 0 || end < start || static_cast<std::size_t>(end) > data_size)
                throw std::runtime_error("invalid offsets for provided utf8_data");

            std::string_view sv(base + start, static_cast<std::size_t>(end - start));
            auto it = vocab_.find(sv);
            out_buf(i) = (it == vocab_.end()) ? -1 : it->second;
        }
        return out;
    }

private:
    phmap::flat_hash_map<std::string, std::int64_t> vocab_;
};

class Uint64Mapper {
public:
    explicit Uint64Mapper(py::dict vocab) {
        for (auto kv : vocab) {
            std::uint64_t key = static_cast<std::uint64_t>(py::cast<std::uint64_t>(kv.first));
            std::int64_t  val = static_cast<std::int64_t>(py::cast<std::int64_t>(kv.second));
            vocab_[key] = val;
        }
    }

    // Map a uint64 NumPy array to int64 IDs using the internal hash map. Unknowns → -1.
    py::array_t<std::int64_t> map_ids(
        py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> ids) const {

        auto in = ids.unchecked<1>();
        py::array_t<std::int64_t> out(in.shape(0));
        auto out_buf = out.mutable_unchecked<1>();

        for (ssize_t i = 0; i < in.shape(0); ++i) {
            auto it = vocab_.find(in(i));
            out_buf(i) = (it == vocab_.end()) ? -1 : it->second;
        }
        return out;
    }

private:
    phmap::flat_hash_map<std::uint64_t, std::int64_t> vocab_;
};

} // namespace fastgrams

// PyBind glue
PYBIND11_MODULE(_fastgrams, m) {
    m.doc() = "Fast mapper utilities.";

    py::class_<fastgrams::ArrowStringArrayMapper>(m, "ArrowStringArrayMapper")
        .def(py::init<py::dict>(), py::arg("vocab"))
        .def("map_ids", &fastgrams::ArrowStringArrayMapper::map_ids,
             py::arg("utf8_data"), py::arg("offsets"));

    py::class_<fastgrams::Uint64Mapper>(m, "Uint64Mapper")
        .def(py::init<py::dict>(), py::arg("vocab"))
        .def("map_ids", &fastgrams::Uint64Mapper::map_ids,
             py::arg("ids"));
} 