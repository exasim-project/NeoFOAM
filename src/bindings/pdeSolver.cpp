// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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

// PDE<Vec3>'s linear system (segregated vector-solve form: scalar matrix, Vec3 rhs).
using Vec3PdeLinearSystem = NeoN::la::LinearSystem<NeoN::scalar, NeoN::Vec3>;

namespace NeoFOAM::bindings
{

void registerPDESolver(nb::module_& m)
{
    // -------------------------------------------------------------------
    // DdtScheme enum (NeoN)
    // -------------------------------------------------------------------
    nb::enum_<fvcc::DdtScheme>(m, "DdtScheme")
        .value("None", fvcc::DdtScheme::None)
        .value("SteadyState", fvcc::DdtScheme::SteadyState)
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
            // keep_alive<1,2>: ``expr`` is taken by value, so nanobind drops the Python
            // Expression once this returns — and with it the chain of keep_alives holding
            // any field a contribution built on the fly (e.g. cross(omega, U)). The
            // operators inside the copied Expression still reference those fields.
            nb::keep_alive<1, 2>(),
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
        )
        .def(
            "relax",
            &nf::PDE<NeoN::scalar>::relax,
            "on"_a = true,
            "Request relaxationFactors.equations relaxation (fvMatrix::relax(); off by default)"
        )
        .def(
            "linear_system",
            // PDE::LinearSystem is a private nested alias; name the concrete type directly.
            static_cast<NeoN::la::LinearSystem<NeoN::scalar>& (nf::PDE<NeoN::scalar>::*)()>(
                &nf::PDE<NeoN::scalar>::linearSystem
            ),
            nb::rv_policy::reference_internal,
            "Get the assembled linear system (matrix + rhs); call assemble() or solve() first"
        );

    m.def(
        "set_scalar_field_values",
        [](fvcc::VolumeField<NeoN::scalar>& field, std::vector<NeoN::scalar> values)
        {
            // Debug-only (investigation_summary.md §4.35): overwrite a scalar field's
            // internal values from a plain Python list -- used to write back an exact
            // direct solve (scipy spsolve) result in place of the iterative solver's
            // own output, to test whether the iterative solver itself is responsible
            // for the NEORHO_ZERO_PDDT_PSI ablation's divergence.
            field.internalVector() = NeoN::Vector<NeoN::scalar>(field.exec(), values);
        },
        "field"_a,
        "values"_a,
        "Debug: overwrite a scalar VolumeField's internal values from a plain Python list"
    );

    m.def(
        "set_scalar_field_boundary_values",
        [](fvcc::VolumeField<NeoN::scalar>& field, std::vector<NeoN::scalar> values)
        {
            // he/T boundary-mapping path (investigation_summary.md): write a live,
            // per-iteration-recomputed boundary value into a scalar field -- mirrors
            // literal OpenFOAM's fixedEnergy/gradientEnergy BCs, which recompute he's
            // boundary value from T's current boundary value every call, not once at
            // setup. Same pattern as set_scalar_field_values above, targeting the
            // boundary storage instead of the internal one.
            field.boundaryData().value() = NeoN::Vector<NeoN::scalar>(field.exec(), values);
        },
        "field"_a,
        "values"_a,
        "Overwrite a scalar VolumeField's boundary values from a plain Python list"
    );

    m.def(
        "create_scalar_field_with_boundary_types",
        [](nf::RunTime& rt, std::string name, std::vector<std::string> patchTypes)
        {
            const NeoN::Executor& exec = rt.exec;
            const NeoN::UnstructuredMesh& mesh = rt.nfMesh;
            // he/T boundary-mapping (investigation_summary.md): build a field whose patches
            // are FixedEnergy/GradientEnergy/Empty -- one type name per boundary patch, in
            // patch order -- so a solver can construct `he` with the right real NeoN BC
            // objects attached, rather than inheriting T's own (FixedValue/FixedGradient)
            // ones. Mirrors fvcc::createCalculatedBCs' pattern of a one-entry "type" dict per
            // patch, generalised to a caller-supplied type per patch instead of one type for
            // all.
            std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs;
            bcs.reserve(patchTypes.size());
            for (NeoN::localIdx patchID = 0; patchID < static_cast<NeoN::localIdx>(patchTypes.size());
                 ++patchID)
            {
                NeoN::Dictionary patchDict(
                    {{"type", patchTypes[static_cast<size_t>(patchID)]}}
                );
                bcs.emplace_back(mesh, patchDict, patchID);
            }
            return fvcc::VolumeField<NeoN::scalar>(exec, std::move(name), mesh, bcs);
        },
        "rt"_a,
        "name"_a,
        "patch_types"_a,
        "Create a scalar VolumeField whose patches are the given BC type names, one per "
        "boundary patch in patch order (e.g. [\"fixedEnergy\", \"gradientEnergy\", "
        "\"gradientEnergy\", \"empty\"])"
    );

    m.def(
        "map_energy_boundary_from_temperature",
        [](const fvcc::VolumeField<NeoN::scalar>& temperature,
           fvcc::VolumeField<NeoN::scalar>& energy,
           NeoN::scalar Cv,
           NeoN::scalar Tref)
        {
            fvcc::mapEnergyBoundaryFromTemperature(
                temperature.boundaryData(), energy.boundaryData(), Cv, Tref
            );
        },
        "temperature"_a,
        "energy"_a,
        "Cv"_a,
        "Tref"_a,
        "Map a temperature field's boundary (refValue/refGrad/valueFraction) onto an energy "
        "field's boundary via he = Cv*(T - Tref); call energy.correct_boundary_conditions() "
        "afterward to materialise the values"
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
            // keep_alive<1,2>: ``expr`` is taken by value, so nanobind drops the Python
            // Expression once this returns — and with it the chain of keep_alives holding
            // any field a contribution built on the fly (e.g. cross(omega, U)). The
            // operators inside the copied Expression still reference those fields.
            nb::keep_alive<1, 2>(),
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
            "relax",
            &nf::PDE<NeoN::Vec3>::relax,
            "on"_a = true,
            "Request relaxationFactors.equations relaxation (fvMatrix::relax(); off by default)"
        )
        .def(
            "ddt_scheme",
            &nf::PDE<NeoN::Vec3>::ddtScheme,
            "Get the ddt scheme determined from fvSchemes"
        )
        .def(
            "linear_system",
            // PDE::LinearSystem is a private nested alias; name the concrete type directly.
            static_cast<Vec3PdeLinearSystem& (nf::PDE<NeoN::Vec3>::*)()>(
                &nf::PDE<NeoN::Vec3>::linearSystem
            ),
            nb::rv_policy::reference_internal,
            "Get the assembled linear system (matrix + rhs); call assemble() or solve() first"
        );
}

} // namespace NeoFOAM::bindings
