// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>

// NeoN headers
#include "NeoN/NeoN.hpp"

// NeoFOAM headers
#include "NeoFOAM/datastructures/pde.hpp"

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
        .value("Steady", fvcc::DdtScheme::Steady)
        .value("BDF1", fvcc::DdtScheme::BDF1)
        .value("BDF2", fvcc::DdtScheme::BDF2);

    // -------------------------------------------------------------------
    // PDE<scalar>
    // -------------------------------------------------------------------
    nb::class_<nf::PDE<NeoN::scalar>>(m, "PDESolverScalar")
        .def(
            "__init__",
            [](nf::PDE<NeoN::scalar>& self,
               dsl::Expression<NeoN::scalar> expr,
               fvcc::VolumeField<NeoN::scalar>& psi,
               nf::RunTime& rt) { new (&self) nf::PDE<NeoN::scalar>(std::move(expr), psi, rt); },
            "expr"_a,
            "psi"_a,
            "runtime"_a,
            nb::keep_alive<1, 3>(),
            nb::keep_alive<1, 4>()
        )
        .def(
            "solve",
            [](nf::PDE<NeoN::scalar>& self) { return self.solve(); },
            "Solve the linear system"
        )
        .def(
            "assemble",
            [](nf::PDE<NeoN::scalar>& self) -> void { self.assemble(); },
            "Assemble the linear system"
        )
        .def(
            "set_reference",
            &nf::PDE<NeoN::scalar>::setReference,
            "ref_cell"_a,
            "ref_value"_a,
            "Set pressure reference cell and value"
        )
        .def(
            "set_final_iter",
            &nf::PDE<NeoN::scalar>::setFinalIter,
            "final_iter"_a,
            "Select the <field>Final solver subdict / relaxation on the final outer pass"
        );

    // -------------------------------------------------------------------
    // PDE<Vec3>
    // -------------------------------------------------------------------
    nb::class_<nf::PDE<NeoN::Vec3>>(m, "PDESolverVec3")
        .def(
            "__init__",
            [](nf::PDE<NeoN::Vec3>& self,
               dsl::Expression<NeoN::Vec3> expr,
               fvcc::VolumeField<NeoN::Vec3>& psi,
               nf::RunTime& rt) { new (&self) nf::PDE<NeoN::Vec3>(std::move(expr), psi, rt); },
            "expr"_a,
            "psi"_a,
            "runtime"_a,
            nb::keep_alive<1, 3>(),
            nb::keep_alive<1, 4>()
        )
        .def(
            "solve",
            [](nf::PDE<NeoN::Vec3>& self) { return self.solve(); },
            "Solve the linear system"
        )
        .def(
            "solve_with_source",
            [](nf::PDE<NeoN::Vec3>& self, dsl::SpatialOperator<NeoN::Vec3> rhs)
            { return self.solve(std::move(rhs)); },
            "rhs"_a,
            "Solve with an explicit source term (e.g. -grad(p))"
        )
        .def(
            "assemble",
            [](nf::PDE<NeoN::Vec3>& self) -> void { self.assemble(); },
            "Assemble the linear system"
        )
        .def(
            "assemble_and_relax",
            [](nf::PDE<NeoN::Vec3>& self) -> void { self.assembleAndRelax(); },
            "Assemble and apply equation relaxation without solving (no momentum predictor)"
        )
        .def(
            "set_final_iter",
            &nf::PDE<NeoN::Vec3>::setFinalIter,
            "final_iter"_a,
            "Select the <field>Final solver subdict / relaxation on the final outer pass"
        )
        .def(
            "ddt_scheme",
            &nf::PDE<NeoN::Vec3>::ddtScheme,
            "Get the ddt scheme determined from fvSchemes"
        );
}

} // namespace NeoFOAM::bindings
