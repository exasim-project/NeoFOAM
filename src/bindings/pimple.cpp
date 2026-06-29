// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Bindings used by the Python neoPimpleFoam port (the NeoN-based PIMPLE solver).
// These mirror the C++ helpers exercised by examples/neoPimpleFoam/neoPimpleFoam.cpp
// and test/pimpleParity.cpp that were not yet exposed to Python:
//   - the outer PimpleControl loop (residual-driven nOuterCorrectors),
//   - the explicit dev2 viscous-stress term (laminar nuEff = nu + nut, nut = 0),
//   - the Gauss-Green velocity gradient tensor it consumes,
//   - explicit field under-relaxation of p between outer correctors,
//   - the continuity-error report, and a uniform volume scalar field factory.

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

// NeoN headers
#include "NeoN/NeoN.hpp"
#include "NeoN/dsl/solver.hpp" // fieldRelaxationSnapshot / applyFieldRelaxation
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp" // GaussGreenGrad

// NeoFOAM headers
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoFOAM/solutionControl/pimpleControl.hpp"
#include "NeoFOAM/fvcc/operators/viscousStressOperator.hpp"
#include "NeoFOAM/turbulenceModels/turbulenceModel.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"
#include "NeoFOAM/auxiliary/continuityError.hpp"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace
{
// Move-only owning handle around the polymorphic TurbulenceModel, so the factory
// returns a single concrete (non-polymorphic) type — nanobind then needs neither
// the abstract base's construct slots nor the concrete models registered.
struct TurbulenceModelHandle
{
    std::unique_ptr<nf::TurbulenceModel> model;

    void validate(const fvcc::VolumeField<NeoN::Vec3>& U) { model->validate(U); }
    void correct(
        const fvcc::VolumeField<NeoN::Vec3>& U,
        fvcc::SurfaceField<NeoN::scalar>& phi,
        nf::RunTime& rt
    )
    {
        model->correct(U, phi, rt);
    }
    fvcc::SurfaceField<NeoN::scalar>& nuEff() { return model->nuEff(); }
    const fvcc::VolumeField<NeoN::scalar>& nut() const { return model->nut(); }
    const fvcc::VolumeField<NeoN::Tensor>& gradU() const { return model->gradU(); }
    void rotateOldTimes() { model->rotateOldTimes(); }
    void write(nf::MeshAdapter& mesh) const { model->write(mesh); }
};
} // namespace

namespace NeoFOAM::bindings
{

void registerPimple(nb::module_& m)
{
    // -------------------------------------------------------------------
    // VolumeField<Tensor> — opaque handle so Python can hold the velocity
    // gradient between gradTensor() and the viscousStress() operator.
    // -------------------------------------------------------------------
    nb::class_<fvcc::VolumeField<NeoN::Tensor>>(m, "TensorVolumeField");

    // -------------------------------------------------------------------
    // Uniform volume scalar field (nu, nut=0) for the explicit viscous stress.
    // Mirrors create_uniform_surface_field but on a VolumeField.
    // -------------------------------------------------------------------
    m.def(
        "create_uniform_volume_field",
        [](nf::RunTime& rt, const std::string& name, double value)
        {
            auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(rt.nfMesh);
            fvcc::VolumeField<NeoN::scalar> field(rt.exec, name, rt.nfMesh, bcs);
            NeoN::fill(field.internalVector(), value);
            NeoN::fill(field.boundaryData().value(), value);
            return field;
        },
        "runtime"_a,
        "name"_a,
        "value"_a,
        "Create a uniform scalar volume field (e.g. nu or nut=0)"
    );

    // -------------------------------------------------------------------
    // Gauss-Green gradient operator — gradTensor(U) -> VolumeField<Tensor>.
    // -------------------------------------------------------------------
    nb::class_<fvcc::GaussGreenGrad>(m, "GaussGreenGrad")
        .def(
            "__init__",
            [](fvcc::GaussGreenGrad* self, nf::RunTime& rt)
            { new (self) fvcc::GaussGreenGrad(rt.exec, rt.nfMesh); },
            "runtime"_a,
            nb::keep_alive<1, 2>(), // the mesh reference must outlive the operator
            "Construct a Gauss-Green gradient operator on the runtime mesh"
        )
        .def(
            "grad_tensor",
            [](fvcc::GaussGreenGrad& self, const fvcc::VolumeField<NeoN::Vec3>& u)
            { return self.gradTensor(u); },
            "u"_a,
            "Compute the velocity gradient tensor field grad(U)"
        );

    // -------------------------------------------------------------------
    // Explicit dev2 viscous-stress operator: div(nuEff*dev2(T(grad(U)))).
    // Returns a NeoN SpatialOperator<Vec3> that composes into the momentum
    // expression. The operator stores references to nu/nut/gradU, so the
    // caller must keep them alive until the equation is assembled.
    // -------------------------------------------------------------------
    m.def(
        "viscous_stress",
        [](const fvcc::VolumeField<NeoN::scalar>& nu,
           const fvcc::VolumeField<NeoN::scalar>& nut,
           const fvcc::VolumeField<NeoN::Tensor>& gradU)
        { return nf::makeViscousStress(nu, nut, gradU); },
        "nu"_a,
        "nut"_a,
        "grad_u"_a,
        nb::keep_alive<0, 1>(),
        nb::keep_alive<0, 2>(),
        nb::keep_alive<0, 3>(),
        "Explicit viscous stress div(nuEff*dev2(T(grad(U)))) for laminar/RAS momentum"
    );

    // -------------------------------------------------------------------
    // Turbulence model (runtime-selected from constant/turbulenceProperties:
    // laminar or LES SpalartAllmarasDDES). Owns nut/nuTilda/gradU internally.
    // -------------------------------------------------------------------
    nb::class_<TurbulenceModelHandle>(m, "TurbulenceModel")
        .def(
            "validate",
            &TurbulenceModelHandle::validate,
            "u"_a,
            "Seed gradU and nut before the time loop"
        )
        .def(
            "correct",
            &TurbulenceModelHandle::correct,
            "u"_a,
            "phi"_a,
            "runtime"_a,
            "Update the turbulence model after the PIMPLE loop each time step"
        )
        .def(
            "nu_eff",
            &TurbulenceModelHandle::nuEff,
            nb::rv_policy::reference_internal,
            "Surface effective viscosity (nu + nut) for the momentum laplacian"
        )
        .def(
            "nut",
            &TurbulenceModelHandle::nut,
            nb::rv_policy::reference_internal,
            "Volume turbulent viscosity for the explicit viscousStress term"
        )
        .def(
            "grad_u",
            &TurbulenceModelHandle::gradU,
            nb::rv_policy::reference_internal,
            "Velocity gradient tensor updated each correct() call"
        )
        .def("rotate_old_times", &TurbulenceModelHandle::rotateOldTimes)
        .def("write", &TurbulenceModelHandle::write, "mesh"_a, "Write model-owned fields");

    m.def(
        "create_turbulence_model",
        [](nf::RunTime& rt, const fvcc::VolumeField<NeoN::scalar>& nu)
        { return TurbulenceModelHandle {nf::createTurbulenceModel(rt, nu)}; },
        "runtime"_a,
        "nu"_a,
        nb::keep_alive<0, 1>(),
        nb::keep_alive<0, 2>(),
        "Create the turbulence model selected in constant/turbulenceProperties"
    );

    // -------------------------------------------------------------------
    // Outer PIMPLE loop control (NeoFOAM-native, residual-driven).
    // residuals is a {field: (initResNorm, finalResNorm)} dict fed each pass.
    // -------------------------------------------------------------------
    nb::class_<nf::PimpleControl>(m, "PimpleControl")
        .def(
            "__init__",
            [](nf::PimpleControl* self, const NeoN::Dictionary& fvSolution)
            { new (self) nf::PimpleControl(fvSolution); },
            "fv_solution"_a,
            "Construct from the fvSolution dict (reads the PIMPLE subdict)"
        )
        .def(
            "loop",
            &nf::PimpleControl::loop,
            "residuals"_a,
            "Advance the outer corrector; returns True while another pass should run"
        )
        .def("final_iter", &nf::PimpleControl::finalIter, "True on the last outer pass")
        .def("first_iter", &nf::PimpleControl::firstIter, "True on the first outer pass")
        .def("n_outer_correctors", &nf::PimpleControl::nOuterCorrectors);

    // -------------------------------------------------------------------
    // Explicit field under-relaxation of p between outer correctors.
    // -------------------------------------------------------------------
    m.def(
        "field_relaxation_snapshot",
        [](const fvcc::VolumeField<NeoN::scalar>& field)
        { return NeoN::dsl::fieldRelaxationSnapshot(field); },
        "field"_a,
        "Deep-copy a field's internal vector for use as the relaxation reference"
    );

    m.def(
        "apply_field_relaxation",
        [](fvcc::VolumeField<NeoN::scalar>& solution,
           const NeoN::Vector<NeoN::scalar>& previous,
           double alpha) { NeoN::dsl::applyFieldRelaxation(solution, previous, alpha); },
        "solution"_a,
        "previous"_a,
        "alpha"_a,
        "Blend solution = previous + alpha*(solution - previous) in place"
    );

    m.def(
        "lookup_field_relaxation",
        [](const NeoN::Dictionary& fvSolution, const std::string& field, bool finalIter) -> double
        { return nf::lookupFieldRelaxation(fvSolution, field, finalIter).value_or(1.0); },
        "fv_solution"_a,
        "field"_a,
        "final_iter"_a,
        "Field under-relaxation factor from relaxationFactors.fields (1.0 if unset)"
    );

    // -------------------------------------------------------------------
    // Continuity-error report (sum local, global) — matches continuityErrs.H.
    // -------------------------------------------------------------------
    m.def(
        "compute_continuity_error",
        [](const fvcc::SurfaceField<NeoN::scalar>& phi, const nf::RunTime& rt)
        {
            const auto errs = nf::computeContinuityError(phi, rt);
            return std::make_pair(errs.sumLocal, errs.global);
        },
        "phi"_a,
        "runtime"_a,
        "Return (sum_local, global) time-step continuity errors from the face flux"
    );
}

} // namespace NeoFOAM::bindings
