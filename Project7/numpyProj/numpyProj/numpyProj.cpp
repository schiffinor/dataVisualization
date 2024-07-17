#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

py::array_t<double> compute_abs_difference(const py::array_t<double>& a, const py::array_t<double>& b) {
    const auto buf_a = a.request(), buf_b = b.request();
    if (buf_a.ndim != 2 || buf_b.ndim != 2) {
        throw std::runtime_error("Input should be 2-D NumPy arrays");
    }
    if (buf_a.shape[1] != buf_b.shape[1]) {
        throw std::runtime_error("The inner dimensions must match");
    }

    Py_ssize_t rows_a = buf_a.shape[0], cols_a = buf_a.shape[1];
    Py_ssize_t rows_b = buf_b.shape[0];

    auto result = py::array_t<double>({rows_a, rows_b, cols_a});
    const auto buf_result = result.request();

    const auto *a_ptr = static_cast<double *>(buf_a.ptr);
    const auto *b_ptr = static_cast<double *>(buf_b.ptr);
    const auto result_ptr = static_cast<double *>(buf_result.ptr);

#pragma omp parallel for
    for (Py_ssize_t i = 0; i < rows_a; ++i) {
        for (Py_ssize_t j = 0; j < rows_b; ++j) {
            for (Py_ssize_t k = 0; k < cols_a; ++k) {
                result_ptr[i * rows_b * cols_a + j * cols_a + k] = std::abs(a_ptr[i * cols_a + k] - b_ptr[j * cols_a + k]);
            }
        }
    }

    return result;
}

PYBIND11_MODULE(numpyProj, m) {
    m.doc() = "A module that computes absolute differences between numpy arrays";
    m.def("compute_abs_difference", &compute_abs_difference, "Compute the absolute differences between two ndarrays");
}
