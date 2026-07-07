// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>

#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/auxiliary/setup.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

void registerRuntime(nb::module_& m)
{
    // -------------------------------------------------------------------
    // MeshAdapter — opaque wrapper so Python can hold a reference
    // -------------------------------------------------------------------
    nb::class_<nf::MeshAdapter>(m, "MeshAdapter");

    // -------------------------------------------------------------------
    // RunTime Struct (Adapter)
    // -------------------------------------------------------------------
    nb::class_<nf::RunTime>(m, "RunTime")
        .def_prop_ro("db", [](nf::RunTime& self) -> NeoN::Database& { return self.db; })
        .def_prop_ro(
            "mesh",
            [](nf::RunTime& self) -> nf::MeshAdapter& { return self.mesh; },
            nb::rv_policy::reference_internal
        )
        .def_prop_ro(
            "nf_mesh",
            [](nf::RunTime& self) -> const NeoN::UnstructuredMesh& { return self.nfMesh; },
            nb::rv_policy::reference_internal
        )
        .def_prop_ro("executor", [](nf::RunTime& self) -> NeoN::Executor { return self.exec; })
        .def_rw("t", &nf::RunTime::t)
        .def_rw("dt", &nf::RunTime::dt)
        .def_rw("adjust_time_step", &nf::RunTime::adjustTimeStep)
        .def_rw("max_co", &nf::RunTime::maxCo)
        .def_rw("max_delta_t", &nf::RunTime::maxDeltaT)
        .def_rw("fv_solution_dict", &nf::RunTime::fvSolutionDict)
        .def_rw("fv_schemes_dict", &nf::RunTime::fvSchemesDict);

    m.def(
        "create_adapter_run_time",
        [](const Foam::Time& rt, const std::string& executor) -> nf::RunTime
        {
            // Build the executor by name (Serial/CPU/GPU/default) with the default
            // allocator. This avoids the strict controlDict "executor"/"allocator"
            // lookups of the 1-arg createAdapterRunTime, so a stock pimpleFoam case
            // (no NeoN executor keys) works. Default Serial matches the deterministic
            // parity setup in test/pimpleParity.cpp.
            auto exec = nf::createExecutor(Foam::word(executor));
            return nf::createAdapterRunTime(rt, exec);
        },
        "rt"_a,
        "executor"_a = std::string("Serial"),
        "Create a NeoFOAM RunTime from an OpenFOAM Time object (executor by name)"
    );

    m.def(
        "sync_run_times",
        &nf::syncRunTimes,
        "of_run_time"_a,
        "nf_run_time"_a,
        "co_num"_a,
        "Sync between OpenFOAM and NeoFOAM runtimes"
    );
}

} // namespace NeoFOAM::bindings
