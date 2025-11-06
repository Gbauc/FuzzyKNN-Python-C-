#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "knnfuzzy.hpp" 

namespace py = pybind11;

// Função auxiliar para converter numpy.ndarray -> std::vector<std::vector<T>>
template <typename T>
std::vector<std::vector<T>> numpy_to_vector(py::array_t<T> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Esperado array 2D");

    std::vector<std::vector<T>> result(buf.shape[0], std::vector<T>(buf.shape[1]));
    T* ptr = static_cast<T*>(buf.ptr);

    for (ssize_t i = 0; i < buf.shape[0]; ++i)
        for (ssize_t j = 0; j < buf.shape[1]; ++j)
            result[i][j] = ptr[i * buf.shape[1] + j];

    return result;
}

PYBIND11_MODULE(fuzzy_knn, m) {
    m.doc() = "Fuzzy KNN implementation (C++ / pybind11)";

    py::class_<KNNResult<double>>(m, "KNNResult")
        .def_readonly("predictions", &KNNResult<double>::predictions)
        .def_readonly("memberships", &KNNResult<double>::memberships);

    py::class_<KNN<double>>(m, "KNN")
        .def(py::init<int, double>(), py::arg("k_neighbors"), py::arg("fuzziness") = 2.0)
        .def("_fit",
            [](KNN<double>& self, py::object x_train, py::object y_train) {
                std::vector<std::vector<double>> x_vec;
                std::vector<int> y_vec;

                if (py::isinstance<py::array>(x_train))
                    x_vec = numpy_to_vector<double>(x_train.cast<py::array_t<double>>());
                else
                    x_vec = x_train.cast<std::vector<std::vector<double>>>();

                if (py::isinstance<py::array>(y_train))
                    y_vec = std::vector<int>(y_train.cast<py::array_t<int>>().data(),
                                             y_train.cast<py::array_t<int>>().data() + y_train.cast<py::array_t<int>>().size());
                else
                    y_vec = y_train.cast<std::vector<int>>();

                self._fit(x_vec, y_vec);
            })
        .def("_predict",
            [](KNN<double>& self, py::object x_test) {
                std::vector<std::vector<double>> x_vec;
                if (py::isinstance<py::array>(x_test))
                    x_vec = numpy_to_vector<double>(x_test.cast<py::array_t<double>>());
                else
                    x_vec = x_test.cast<std::vector<std::vector<double>>>();

                return self._predict(x_vec);
            });
}

