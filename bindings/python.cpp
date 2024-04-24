// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "python.hpp"

namespace py = pybind11;


PYBIND11_MODULE(pyNeoFOAM, m)
{
    m.doc() = "Python bindings for the NeoFOAM framework";

    py::class_<NeoFOAM::CPUExecutor>(m, "CPUExecutor")
        .def(py::init<>());

    py::class_<NeoFOAM::OMPExecutor>(m, "OMPExecutor")
        .def(py::init<>());

    py::class_<NeoFOAM::GPUExecutor>(m, "GPUExecutor")
        .def(py::init<>());

    // py::class_<NeoFOAM::Field<double>>(m, "Field")
    //    .def(py::init<NeoFOAM::CPUExecutor&, size_t>())
    //    .def("fill", &NeoFOAM::Field<double>::operator=, "Fill the Field with given value");
}
