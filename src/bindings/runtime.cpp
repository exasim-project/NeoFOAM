// SPDX-FileCopyrightText: 2024-2026 NeoFOAM authors
// SPDX-License-Identifier: Unlicense

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>

#include "NeoFOAM/datastructures/runTime.hpp"
#include "runtime.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

void registerRuntime(nb::module_& m)
{
    // -------------------------------------------------------------------
    // Runtime
    // -------------------------------------------------------------------
    nb::class_<Runtime>(m, "Runtime")
        .def(
            nb::init<std::vector<std::string>>(),
            "argv"_a,
            "Create runtime from command-line arguments (creates OF Time + NeoN mesh)"
        )
        // Time loop
        .def("loop", &Runtime::loop, "Advance time loop; False when finished")
        .def("time", &Runtime::time)
        .def("delta_t", &Runtime::deltaT)
        .def("time_name", &Runtime::timeName)
        .def("sync", &Runtime::sync, "co_num"_a, "Sync OF <-> NeoN time/deltaT")
        // IO
        .def("write", &Runtime::write)
        .def("output_time", &Runtime::outputTime)
        .def("print_execution_time", &Runtime::printExecutionTime)
        // NeoFOAM RunTime
        .def(
            "nf_runtime",
            [](Runtime& self) -> nf::RunTime& { return self.nfRuntime(); },
            nb::rv_policy::reference_internal
        )
        .def("executor", [](Runtime& self) -> NeoN::Executor { return self.executor(); })
        .def(
            "nf_mesh",
            [](Runtime& self) -> const NeoN::UnstructuredMesh& { return self.nfMesh(); },
            nb::rv_policy::reference_internal
        )
        .def(
            "db",
            [](Runtime& self) -> NeoN::Database& { return self.db(); },
            nb::rv_policy::reference_internal
        );

    // -------------------------------------------------------------------
    // PisoControl
    // -------------------------------------------------------------------
    nb::class_<PisoControl>(m, "PisoControl")
        .def(nb::init<Runtime&>(), "runtime"_a, nb::keep_alive<1, 2>())
        .def("momentum_predictor", &PisoControl::momentumPredictor)
        .def("correct", &PisoControl::correct)
        .def("correct_non_orthogonal", &PisoControl::correctNonOrthogonal)
        .def("final_non_orthogonal_iter", &PisoControl::finalNonOrthogonalIter);

    // -------------------------------------------------------------------
    // RunTime (read-only NeoFOAM struct, needed by PDESolver)
    // -------------------------------------------------------------------
    nb::class_<nf::RunTime>(m, "RunTime")
        .def_ro("t", &nf::RunTime::t)
        .def_ro("dt", &nf::RunTime::dt);
}

} // namespace NeoFOAM::bindings
