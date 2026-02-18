// SPDX-FileCopyrightText: 2024-2026 NeoFOAM authors
// SPDX-License-Identifier: Unlicense

#include <nanobind/nanobind.h>

// NeoN headers
#include "NeoN/NeoN.hpp"

// NeoFOAM headers
#include "NeoFOAM/datastructures/pdeSolver.hpp"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;
namespace dsl = NeoN::dsl;

namespace NeoFOAM::bindings
{

void registerPDESolver(nb::module_& m)
{
    // -------------------------------------------------------------------
    // DdtScheme enum (NeoN)
    // -------------------------------------------------------------------
    nb::enum_<fvcc::DdtScheme>(m, "DdtScheme")
        .value("None", fvcc::DdtScheme::None)
        .value("BDF1", fvcc::DdtScheme::BDF1)
        .value("BDF2", fvcc::DdtScheme::BDF2);

    // -------------------------------------------------------------------
    // PDESolver<scalar>
    // -------------------------------------------------------------------
    nb::class_<nf::PDESolver<NeoN::scalar>>(m, "PDESolverScalar")
        .def(
            "__init__",
            [](nf::PDESolver<NeoN::scalar>& self,
               dsl::Expression<NeoN::scalar> expr,
               fvcc::VolumeField<NeoN::scalar>& psi,
               const nf::RunTime& rt)
            { new (&self) nf::PDESolver<NeoN::scalar>(std::move(expr), psi, rt); },
            "expr"_a,
            "psi"_a,
            "runtime"_a,
            nb::keep_alive<1, 3>(),
            nb::keep_alive<1, 4>()
        )
        .def(
            "solve",
            [](nf::PDESolver<NeoN::scalar>& self) { return self.solve(); },
            "Solve the linear system"
        )
        .def(
            "assemble",
            [](nf::PDESolver<NeoN::scalar>& self) -> void { self.assemble(); },
            "Assemble the linear system"
        )
        .def(
            "set_reference",
            &nf::PDESolver<NeoN::scalar>::setReference,
            "ref_cell"_a,
            "ref_value"_a,
            "Set pressure reference cell and value"
        );

    // -------------------------------------------------------------------
    // PDESolver<Vec3>
    // -------------------------------------------------------------------
    nb::class_<nf::PDESolver<NeoN::Vec3>>(m, "PDESolverVec3")
        .def(
            "__init__",
            [](nf::PDESolver<NeoN::Vec3>& self,
               dsl::Expression<NeoN::Vec3> expr,
               fvcc::VolumeField<NeoN::Vec3>& psi,
               const nf::RunTime& rt)
            { new (&self) nf::PDESolver<NeoN::Vec3>(std::move(expr), psi, rt); },
            "expr"_a,
            "psi"_a,
            "runtime"_a,
            nb::keep_alive<1, 3>(),
            nb::keep_alive<1, 4>()
        )
        .def(
            "solve",
            [](nf::PDESolver<NeoN::Vec3>& self) { return self.solve(); },
            "Solve the linear system"
        )
        .def(
            "solve_with_source",
            [](nf::PDESolver<NeoN::Vec3>& self, dsl::SpatialOperator<NeoN::Vec3> rhs)
            { return self.solve(std::move(rhs)); },
            "rhs"_a,
            "Solve with an explicit source term (e.g. -grad(p))"
        )
        .def(
            "assemble",
            [](nf::PDESolver<NeoN::Vec3>& self) -> void { self.assemble(); },
            "Assemble the linear system"
        )
        .def(
            "ddt_scheme",
            &nf::PDESolver<NeoN::Vec3>::ddtScheme,
            "Get the ddt scheme determined from fvSchemes"
        );
}

} // namespace NeoFOAM::bindings
