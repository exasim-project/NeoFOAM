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
#include <nanobind/ndarray.h> // copy_from_host (host array -> field)
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

// NeoN headers
#include "NeoN/NeoN.hpp"
#include "NeoN/dsl/solver.hpp" // fieldRelaxationSnapshot / applyFieldRelaxation
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp" // GaussGreenGrad

// OpenFOAM headers
#include "wallDist.H" // Foam::wallDist for read_wall_distance

// NeoFOAM headers
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoFOAM/datastructures/pde.hpp" // PDE<scalar> for the epsilon wall-cell pin
#include "NeoFOAM/fvcc/boundary/volume/omegaWallFunction.hpp" // OMEGA_WF_OMEGA_MAX for the omega pin
#include "NeoFOAM/solutionControl/pimpleControl.hpp"
#include "NeoFOAM/fvcc/operators/viscousStressOperator.hpp"
#include "NeoFOAM/turbulenceModels/turbulenceModel.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"
#include "NeoFOAM/auxiliary/continuityError.hpp"
#include "NeoFOAM/auxiliary/readers.hpp" // constructFrom (Foam field -> NeoN field)

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using NeoN::localIdx;
using NeoN::scalar;

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

// Turbulence production per unit eddy viscosity, G/nut = dev(twoSymm(gradU)) && gradU.
// Kept in a free function: NEON_LAMBDA is an extended __device__ lambda under CUDA,
// which nvcc forbids from being defined inside another lambda (the binding lambda).
fvcc::VolumeField<NeoN::scalar> strainProduction(const fvcc::VolumeField<NeoN::Tensor>& gradU)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(gradU.mesh());
    fvcc::VolumeField<NeoN::scalar> gByNu(gradU.exec(), "GbyNu", gradU.mesh(), bcs);
    auto gv = gradU.internalVector().view();
    auto ov = gByNu.internalVector().view();
    NeoN::parallelFor(
        gByNu.exec(),
        {0, gByNu.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) {
            const NeoN::Tensor t = gv[i];
            const NeoN::Tensor ts = t + t.T(); // twoSymm(gradU)
            const NeoN::scalar third = ts.trace() / NeoN::scalar(3.0);
            NeoN::scalar sum = 0.0;
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    // dev(ts)(r,c) = ts(r,c) - (tr/3) delta_rc
                    const NeoN::scalar dev = ts(r, c) - (r == c ? third : NeoN::scalar(0.0));
                    sum += dev * t(r, c);
                }
            }
            ov[i] = sum;
        },
        "strainProduction"
    );
    auto gb = gradU.boundaryData().value().view();
    auto ob = gByNu.boundaryData().value().view();
    NeoN::parallelFor(
        gByNu.exec(),
        {0, gByNu.boundaryData().value().size()},
        NEON_LAMBDA(const NeoN::localIdx i) {
            const NeoN::Tensor t = gb[i];
            const NeoN::Tensor ts = t + t.T();
            const NeoN::scalar third = ts.trace() / NeoN::scalar(3.0);
            NeoN::scalar sum = 0.0;
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    const NeoN::scalar dev = ts(r, c) - (r == c ? third : NeoN::scalar(0.0));
                    sum += dev * t(r, c);
                }
            }
            ob[i] = sum;
        },
        "strainProduction::boundary"
    );
    return gByNu;
}

// Vorticity magnitude Omega = sqrt(2) * mag(skew(gradU)), the strain invariant the
// Spalart-Allmaras production term Stilda is built from. Free function for the same
// nvcc/NEON_LAMBDA reason as strainProduction.
fvcc::VolumeField<NeoN::scalar> vorticityMagnitude(const fvcc::VolumeField<NeoN::Tensor>& gradU)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(gradU.mesh());
    fvcc::VolumeField<NeoN::scalar> omega(gradU.exec(), "vorticityMagnitude", gradU.mesh(), bcs);
    auto gv = gradU.internalVector().view();
    auto ov = omega.internalVector().view();
    NeoN::parallelFor(
        omega.exec(),
        {0, omega.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) {
            const NeoN::Tensor t = gv[i];
            NeoN::scalar s = 0.0; // sum of squares of skew(gradU) components
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    const NeoN::scalar sk = NeoN::scalar(0.5) * (t(r, c) - t(c, r));
                    s += sk * sk;
                }
            }
            // Omega = sqrt(2)*mag(skew) = sqrt(2 * sum(skew^2))
            ov[i] = Kokkos::sqrt(NeoN::scalar(2.0) * s);
        },
        "vorticityMagnitude"
    );
    auto gb = gradU.boundaryData().value().view();
    auto ob = omega.boundaryData().value().view();
    NeoN::parallelFor(
        omega.exec(),
        {0, omega.boundaryData().value().size()},
        NEON_LAMBDA(const NeoN::localIdx i) {
            const NeoN::Tensor t = gb[i];
            NeoN::scalar s = 0.0;
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    const NeoN::scalar sk = NeoN::scalar(0.5) * (t(r, c) - t(c, r));
                    s += sk * sk;
                }
            }
            ob[i] = Kokkos::sqrt(NeoN::scalar(2.0) * s);
        },
        "vorticityMagnitude::boundary"
    );
    return omega;
}

// magSqr(grad(phi)) for a scalar field — the Spalart-Allmaras Cb2 diffusion source
// term magSqr(grad(nuTilda)). Evaluates the Gauss-Green gradient of phi (carrying
// phi's boundary conditions) and squares its magnitude per cell.
fvcc::VolumeField<NeoN::scalar>
magSqrGrad(nf::RunTime& rt, const fvcc::VolumeField<NeoN::scalar>& phi)
{
    fvcc::GaussGreenGrad grad(rt.exec, rt.nfMesh);
    fvcc::VolumeField<NeoN::Vec3> gradPhi = grad.grad(phi);
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(rt.nfMesh);
    fvcc::VolumeField<NeoN::scalar> out(rt.exec, "magSqrGrad", rt.nfMesh, bcs);
    auto gv = gradPhi.internalVector().view();
    auto ov = out.internalVector().view();
    NeoN::parallelFor(
        out.exec(),
        {0, out.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) {
            const NeoN::Vec3 g = gv[i];
            ov[i] = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        },
        "magSqrGrad"
    );
    return out;
}

// Strain-rate magnitude squared S2 = 2 magSqr(symm(gradU)), the kOmegaSST production
// invariant (nut uses sqrt(S2)). symm(T) = (T + T^T)/2; magSqr sums the 9 components.
fvcc::VolumeField<NeoN::scalar> strainMagnitudeSqr(const fvcc::VolumeField<NeoN::Tensor>& gradU)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(gradU.mesh());
    fvcc::VolumeField<NeoN::scalar> out(gradU.exec(), "strainMagnitudeSqr", gradU.mesh(), bcs);
    auto gv = gradU.internalVector().view();
    auto ov = out.internalVector().view();
    NeoN::parallelFor(
        out.exec(),
        {0, out.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) {
            const NeoN::Tensor t = gv[i];
            NeoN::scalar s = 0.0; // sum of squares of symm(gradU) components
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    const NeoN::scalar sy = NeoN::scalar(0.5) * (t(r, c) + t(c, r));
                    s += sy * sy;
                }
            }
            ov[i] = NeoN::scalar(2.0) * s; // 2 magSqr(symm(gradU))
        },
        "strainMagnitudeSqr"
    );
    auto gb = gradU.boundaryData().value().view();
    auto ob = out.boundaryData().value().view();
    NeoN::parallelFor(
        out.exec(),
        {0, out.boundaryData().value().size()},
        NEON_LAMBDA(const NeoN::localIdx i) {
            const NeoN::Tensor t = gb[i];
            NeoN::scalar s = 0.0;
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    const NeoN::scalar sy = NeoN::scalar(0.5) * (t(r, c) + t(c, r));
                    s += sy * sy;
                }
            }
            ob[i] = NeoN::scalar(2.0) * s;
        },
        "strainMagnitudeSqr::boundary"
    );
    return out;
}

// grad(a) & grad(b) for two scalar fields — the kOmegaSST cross-diffusion
// CDkOmega = 2 alphaOmega2 (grad(k) . grad(omega)) / omega.
fvcc::VolumeField<NeoN::scalar> gradDotGrad(
    nf::RunTime& rt,
    const fvcc::VolumeField<NeoN::scalar>& a,
    const fvcc::VolumeField<NeoN::scalar>& b
)
{
    fvcc::GaussGreenGrad grad(rt.exec, rt.nfMesh);
    fvcc::VolumeField<NeoN::Vec3> gradA = grad.grad(a);
    fvcc::VolumeField<NeoN::Vec3> gradB = grad.grad(b);
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(rt.nfMesh);
    fvcc::VolumeField<NeoN::scalar> out(rt.exec, "gradDotGrad", rt.nfMesh, bcs);
    auto av = gradA.internalVector().view();
    auto bv = gradB.internalVector().view();
    auto ov = out.internalVector().view();
    NeoN::parallelFor(
        out.exec(),
        {0, out.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) {
            const NeoN::Vec3 ga = av[i];
            const NeoN::Vec3 gb = bv[i];
            ov[i] = ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2];
        },
        "gradDotGrad"
    );
    return out;
}
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
    // Turbulence production per unit eddy viscosity: dev(twoSymm(gradU)) && gradU,
    // the scalar field G/nut used by two-equation closures (kEpsilon, ...). The
    // tensor contraction lives here (there is no field-level tensor algebra in
    // Python); the closure multiplies the result by nut in readable field maths.
    // -------------------------------------------------------------------
    // -------------------------------------------------------------------
    // Spalart-Allmaras primitives that have no field-level counterpart in the
    // Python DSL: the vorticity magnitude and the scalar-gradient magnitude
    // squared (both small tensor/vector contractions kept in Kokkos kernels),
    // the wall-distance field, and a host->device write so the closure's
    // nonlinear scalar maths (chi, fv1, fw, ...) can be authored in plain NumPy.
    // -------------------------------------------------------------------
    m.def(
        "vorticity_magnitude",
        &vorticityMagnitude,
        "grad_u"_a,
        "Vorticity magnitude Omega = sqrt(2) mag(skew(gradU)) (Spalart-Allmaras)"
    );

    m.def(
        "mag_sqr_grad",
        &magSqrGrad,
        "runtime"_a,
        "phi"_a,
        "magSqr(grad(phi)) for a scalar field (Spalart-Allmaras Cb2 source)"
    );

    m.def(
        "strain_magnitude_sqr",
        &strainMagnitudeSqr,
        "grad_u"_a,
        "Strain-rate magnitude squared S2 = 2 magSqr(symm(gradU)) (kOmegaSST)"
    );

    m.def(
        "grad_dot_grad",
        &gradDotGrad,
        "runtime"_a,
        "a"_a,
        "b"_a,
        "grad(a) . grad(b) for two scalar fields (kOmegaSST cross-diffusion)"
    );

    m.def(
        "read_wall_distance",
        [](nf::RunTime& rt) -> fvcc::VolumeField<NeoN::scalar>
        {
            Foam::wallDist y(rt.mesh);
            return NeoFOAM::constructFrom(rt.exec, rt.nfMesh, y.y());
        },
        "runtime"_a,
        "Wall-distance field y (Foam::wallDist to the nearest wall patch)"
    );

    m.def(
        "copy_from_host",
        [](fvcc::VolumeField<NeoN::scalar>& field,
           nb::ndarray<const NeoN::scalar, nb::ndim<1>, nb::c_contig, nb::device::cpu> values)
        {
            const auto n = field.internalVector().size();
            if (static_cast<std::size_t>(values.shape(0)) != static_cast<std::size_t>(n))
            {
                throw std::runtime_error("copy_from_host: array size != field size");
            }
            field.internalVector() =
                NeoN::Vector<NeoN::scalar>(field.exec(), values.data(), n, NeoN::SerialExecutor());
            field.correctBoundaryConditions();
        },
        "field"_a,
        "values"_a,
        "Overwrite a scalar field's internal values from a host array; corrects BCs"
    );

    m.def(
        "strain_production",
        &strainProduction,
        "grad_u"_a,
        "Scalar production per eddy viscosity dev(twoSymm(gradU)) && gradU (= G/nut)"
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

    // -------------------------------------------------------------------
    // Turbulence wall-function helpers (kEpsilon steady SIMPLE parity).
    // The pure-Python kEpsilon closure owns the transport equations; these expose
    // the *model-side* half of the epsilon/k/nut wall functions it cannot write in
    // the field DSL (boundary-face views, atomic per-cell scatter):
    //   * build_near_wall_dist  — nearWallDist field (boundary faces = owner-cell y),
    //   * correct_scalar_bc_ctx — correctBoundaryConditions with a (U,k,nu,y) context
    //                             so the WF BCs actually set their face values,
    //   * epsilon_wall_production — the near-wall production (G) override the
    //                             epsilonWallFunction depends on (mirrors
    //                             KEpsilon::correct, kEpsilon.cpp).
    // Top-level (neofoam) only; the NeoN submodule is untouched.
    // -------------------------------------------------------------------
    m.def(
        "build_near_wall_dist",
        [](nf::RunTime& rt) -> fvcc::VolumeField<scalar>
        {
            Foam::wallDist y(rt.mesh);
            const auto wallDist = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, y.y());
            fvcc::VolumeField<scalar> nearWallDist(
                rt.exec,
                "nearWallDist",
                rt.nfMesh,
                fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(rt.nfMesh)
            );
            const auto wd = wallDist.internalVector().view();
            const auto owners = rt.nfMesh.boundaryMesh().faceOwners().view();
            auto nwdB = nearWallDist.boundaryData().value().view();
            const auto nBF = static_cast<localIdx>(nwdB.size());
            NeoN::parallelFor(
                rt.exec,
                {0, nBF},
                NEON_LAMBDA(const localIdx i) { nwdB[i] = wd[owners[i]]; },
                "build_near_wall_dist"
            );
            return nearWallDist;
        },
        "runtime"_a,
        "nearWallDist field: boundary faces hold the owner-cell wall distance (WF input)"
    );

    m.def(
        "correct_scalar_bc_ctx",
        [](fvcc::VolumeField<scalar>& target,
           const fvcc::VolumeField<scalar>& k,
           const fvcc::VolumeField<scalar>& nu,
           const fvcc::VolumeField<scalar>& nearWallDist)
        {
            fvcc::BoundaryContext ctx;
            ctx.insert("k", k);
            ctx.insert("nu", nu);
            ctx.insert("nearWallDist", nearWallDist);
            target.correctBoundaryConditions(ctx);
        },
        "target"_a,
        "k"_a,
        "nu"_a,
        "near_wall_dist"_a,
        "correctBoundaryConditions with a (k,nu,nearWallDist) context for the epsilon/nutk/kqR WFs"
    );

    m.def(
        "epsilon_wall_production",
        [](const fvcc::VolumeField<scalar>& G,
           const fvcc::VolumeField<scalar>& epsilon,
           const fvcc::VolumeField<NeoN::Vec3>& U,
           const fvcc::VolumeField<scalar>& k,
           const fvcc::VolumeField<scalar>& nu,
           const fvcc::VolumeField<scalar>& nut,
           const fvcc::VolumeField<scalar>& nearWallDist,
           const nf::RunTime& rt,
           double cmu,
           double kappa,
           const std::string& wall_patch) -> fvcc::VolumeField<scalar>
        {
            // Copy the bulk production; override only the wall cells (the k equation
            // consumes this; the epsilon/omega equation keeps the un-overridden bulk G).
            // Reused for kOmegaSST (wall_patch="omegaWallFunction", cmu=betaStar) — its
            // near-wall G formula is identical to kEpsilon's.
            fvcc::VolumeField<scalar> Gk = G;
            const auto& exec = rt.exec;
            const auto& mesh = rt.nfMesh;
            const scalar cmu25 = Kokkos::pow(static_cast<scalar>(cmu), scalar(0.25));
            const scalar kappa_ = static_cast<scalar>(kappa);
            const auto epsilonBCs = epsilon.boundaryConditions();
            const auto faceOwnersV = mesh.boundaryMesh().faceOwners().view();
            const auto deltaCoeffsV = mesh.boundaryMesh().deltaCoeffs().view();
            const auto uInternalV = U.internalVector().view();
            const auto uBoundaryV = U.boundaryData().value().view();
            const auto nuBoundaryV = nu.boundaryData().value().view();
            const auto nutBoundaryV = nut.boundaryData().value().view();
            const auto yBoundaryV = nearWallDist.boundaryData().value().view();
            const auto kInternalV = k.internalVector().view();
            auto gkV = Gk.internalVector().view();
            const auto nCells = static_cast<localIdx>(mesh.nCells());

            // cornerWeight[c] = 1/(count of epsilonWallFunction faces on cell c), else 0.
            NeoN::Vector<scalar> cornerWeight(exec, nCells, scalar(0));
            auto cwBuild = cornerWeight.view();
            for (localIdx patchID = 0; patchID < static_cast<localIdx>(epsilonBCs.size());
                 ++patchID)
            {
                if (epsilonBCs[static_cast<size_t>(patchID)].name() != wall_patch) continue;
                const auto [start, end] = epsilon.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        Kokkos::atomic_add(&cwBuild[faceOwnersV[i]], scalar(1));
                    },
                    "epsilon_wall_production::countFaces"
                );
            }
            NeoN::parallelFor(
                exec,
                {0, nCells},
                NEON_LAMBDA(const localIdx c) {
                    if (cwBuild[c] > scalar(0)) cwBuild[c] = scalar(1) / cwBuild[c];
                },
                "epsilon_wall_production::invertCount"
            );
            const auto cornerWeightV = cornerWeight.view();

            // Zero bulk production at wall cells before accumulating the WF form.
            NeoN::parallelFor(
                exec,
                {0, nCells},
                NEON_LAMBDA(const localIdx c) {
                    if (cornerWeightV[c] > scalar(0)) gkV[c] = scalar(0);
                },
                "epsilon_wall_production::zeroWall"
            );
            NeoN::fence(exec);

            for (localIdx patchID = 0; patchID < static_cast<localIdx>(epsilonBCs.size());
                 ++patchID)
            {
                if (epsilonBCs[static_cast<size_t>(patchID)].name() != wall_patch) continue;
                const auto [start, end] = epsilon.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        const auto owner = faceOwnersV[i];
                        const scalar cw = cornerWeightV[owner];
                        const NeoN::Vec3 uOwn = uInternalV[owner];
                        const NeoN::Vec3 uWall = uBoundaryV[i];
                        const scalar deltaInv = deltaCoeffsV[i];
                        const NeoN::Vec3 snGradU = (uWall - uOwn) * deltaInv;
                        const scalar magGradUw = NeoN::mag(snGradU);
                        const scalar nuw = nuBoundaryV[i];
                        const scalar nutw = nutBoundaryV[i];
                        const scalar yv = yBoundaryV[i];
                        const scalar kc = Kokkos::max(kInternalV[owner], scalar(0));
                        const scalar gWall =
                            (nutw + nuw) * magGradUw * cmu25 * Kokkos::sqrt(kc) / (kappa_ * yv);
                        Kokkos::atomic_add(&gkV[owner], cw * gWall);
                    },
                    "epsilon_wall_production::gFeedback"
                );
            }
            return Gk;
        },
        "G"_a,
        "epsilon"_a,
        "U"_a,
        "k"_a,
        "nu"_a,
        "nut"_a,
        "near_wall_dist"_a,
        "runtime"_a,
        "cmu"_a = 0.09,
        "kappa"_a = 0.41,
        "wall_patch"_a = "epsilonWallFunction",
        "k-equation production with the epsilon/omega wall-function near-wall G override"
    );

    m.def(
        "pin_epsilon_wall_cells",
        [](nf::PDE<scalar>& epsPde,
           const fvcc::VolumeField<scalar>& epsilon,
           const fvcc::VolumeField<scalar>& k,
           const fvcc::VolumeField<scalar>& nearWallDist,
           const nf::RunTime& rt,
           double cmu,
           double kappa)
        {
            // OpenFOAM's epsilonWallFunction::manipulateMatrix pins the near-wall CELL
            // epsilon to the log-law value (STEPWISE blender, lowRe off):
            //   epsilon0 = Cmu^0.75 k^1.5 / (kappa y),
            // corner-weight-averaged over the wall faces touching each cell. The BC only
            // sets the wall *face*; this is the cell half. Mirrors the omegaWallFunction
            // pin in kOmegaSST::correct.
            const auto& exec = rt.exec;
            const auto& mesh = rt.nfMesh;
            const scalar Cmu75 = Kokkos::pow(static_cast<scalar>(cmu), scalar(0.75));
            const scalar kappa_ = static_cast<scalar>(kappa);
            const auto epsilonBCs = epsilon.boundaryConditions();
            const auto faceOwnersV = mesh.boundaryMesh().faceOwners().view();
            const auto yBoundaryV = nearWallDist.boundaryData().value().view();
            const auto kInternalV = k.internalVector().view();
            const auto nCells = static_cast<localIdx>(mesh.nCells());

            // cornerWeight[c] = 1/(count of epsilonWallFunction faces on cell c), else 0.
            NeoN::Vector<scalar> cornerWeight(exec, nCells, scalar(0));
            auto cwBuild = cornerWeight.view();
            for (localIdx patchID = 0; patchID < static_cast<localIdx>(epsilonBCs.size());
                 ++patchID)
            {
                if (epsilonBCs[static_cast<size_t>(patchID)].name() != "epsilonWallFunction")
                    continue;
                const auto [start, end] = epsilon.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        Kokkos::atomic_add(&cwBuild[faceOwnersV[i]], scalar(1));
                    },
                    "pin_epsilon_wall_cells::countFaces"
                );
            }
            NeoN::parallelFor(
                exec,
                {0, nCells},
                NEON_LAMBDA(const localIdx c) {
                    if (cwBuild[c] > scalar(0)) cwBuild[c] = scalar(1) / cwBuild[c];
                },
                "pin_epsilon_wall_cells::invertCount"
            );
            const auto cornerWeightV = cornerWeight.view();

            NeoN::Vector<scalar> mask(exec, nCells, scalar(0));
            NeoN::Vector<scalar> values(exec, nCells, scalar(0));
            auto maskV = mask.view();
            auto valuesV = values.view();
            NeoN::fence(exec);

            for (localIdx patchID = 0; patchID < static_cast<localIdx>(epsilonBCs.size());
                 ++patchID)
            {
                if (epsilonBCs[static_cast<size_t>(patchID)].name() != "epsilonWallFunction")
                    continue;
                const auto [start, end] = epsilon.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        const auto owner = faceOwnersV[i];
                        const scalar cw = cornerWeightV[owner];
                        const scalar y = yBoundaryV[i];
                        const scalar kc = Kokkos::max(kInternalV[owner], scalar(0));
                        const scalar eLog = Cmu75 * Kokkos::pow(kc, scalar(1.5)) / (kappa_ * y);
                        Kokkos::atomic_add(&valuesV[owner], cw * eLog);
                        maskV[owner] = scalar(1);
                    },
                    "pin_epsilon_wall_cells::accumulate"
                );
            }
            epsPde.setConstraintsOwned(mask, values);
        },
        "epsilon_pde"_a,
        "epsilon"_a,
        "k"_a,
        "near_wall_dist"_a,
        "runtime"_a,
        "cmu"_a = 0.09,
        "kappa"_a = 0.41,
        "Pin the epsilon equation's near-wall cells to the epsilonWallFunction log-law value"
    );

    m.def(
        "correct_scalar_bc_ctx_u",
        [](fvcc::VolumeField<scalar>& target,
           const fvcc::VolumeField<NeoN::Vec3>& U,
           const fvcc::VolumeField<scalar>& nu,
           const fvcc::VolumeField<scalar>& nearWallDist)
        {
            // nutUSpaldingWallFunction reads U (not k): (U,nu,nearWallDist) context.
            fvcc::BoundaryContext ctx;
            ctx.insert("U", U);
            ctx.insert("nu", nu);
            ctx.insert("nearWallDist", nearWallDist);
            target.correctBoundaryConditions(ctx);
        },
        "target"_a,
        "U"_a,
        "nu"_a,
        "near_wall_dist"_a,
        "correctBoundaryConditions with a (U,nu,nearWallDist) context for the nutUSpalding WF"
    );

    m.def(
        "pin_omega_wall_cells",
        [](nf::PDE<scalar>& omegaPde,
           const fvcc::VolumeField<scalar>& omega,
           const fvcc::VolumeField<scalar>& k,
           const fvcc::VolumeField<scalar>& nu,
           const fvcc::VolumeField<scalar>& nearWallDist,
           const nf::RunTime& rt,
           double beta1,
           double cmu,
           double kappa)
        {
            // OpenFOAM's omegaWallFunction::manipulateMatrix pins the near-wall CELL
            // omega to the blended (BINOMIAL n=2) value:
            //   wVis = 6 nu / (beta1 y^2), wLog = sqrt(k) / (Cmu^0.25 kappa y),
            //   omega0 = min(sqrt(wVis^2 + wLog^2), OMEGA_WF_OMEGA_MAX),
            // corner-weight-averaged over the wall faces touching each cell. The BC only
            // sets the wall *face*; this is the cell half (mirrors kOmegaSST::correct).
            const auto& exec = rt.exec;
            const auto& mesh = rt.nfMesh;
            const scalar cmu25 = Kokkos::pow(static_cast<scalar>(cmu), scalar(0.25));
            const scalar beta1_ = static_cast<scalar>(beta1);
            const scalar kappa_ = static_cast<scalar>(kappa);
            const scalar omegaMax = fvcc::volumeBoundary::detail::OMEGA_WF_OMEGA_MAX;
            const auto omegaBCs = omega.boundaryConditions();
            const auto faceOwnersV = mesh.boundaryMesh().faceOwners().view();
            const auto yBoundaryV = nearWallDist.boundaryData().value().view();
            const auto nuBoundaryV = nu.boundaryData().value().view();
            const auto kInternalV = k.internalVector().view();
            const auto nCells = static_cast<localIdx>(mesh.nCells());

            // cornerWeight[c] = 1/(count of omegaWallFunction faces on cell c), else 0.
            NeoN::Vector<scalar> cornerWeight(exec, nCells, scalar(0));
            auto cwBuild = cornerWeight.view();
            for (localIdx patchID = 0; patchID < static_cast<localIdx>(omegaBCs.size()); ++patchID)
            {
                if (omegaBCs[static_cast<size_t>(patchID)].name() != "omegaWallFunction") continue;
                const auto [start, end] = omega.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        Kokkos::atomic_add(&cwBuild[faceOwnersV[i]], scalar(1));
                    },
                    "pin_omega_wall_cells::countFaces"
                );
            }
            NeoN::parallelFor(
                exec,
                {0, nCells},
                NEON_LAMBDA(const localIdx c) {
                    if (cwBuild[c] > scalar(0)) cwBuild[c] = scalar(1) / cwBuild[c];
                },
                "pin_omega_wall_cells::invertCount"
            );
            const auto cornerWeightV = cornerWeight.view();

            NeoN::Vector<scalar> mask(exec, nCells, scalar(0));
            NeoN::Vector<scalar> values(exec, nCells, scalar(0));
            auto maskV = mask.view();
            auto valuesV = values.view();
            NeoN::fence(exec);

            for (localIdx patchID = 0; patchID < static_cast<localIdx>(omegaBCs.size()); ++patchID)
            {
                if (omegaBCs[static_cast<size_t>(patchID)].name() != "omegaWallFunction") continue;
                const auto [start, end] = omega.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        const auto owner = faceOwnersV[i];
                        const scalar cw = cornerWeightV[owner];
                        const scalar y = yBoundaryV[i];
                        const scalar nuw = nuBoundaryV[i];
                        const scalar kc = Kokkos::max(kInternalV[owner], scalar(0));
                        const scalar wVis = scalar(6) * nuw / (beta1_ * y * y);
                        const scalar wLog = Kokkos::sqrt(kc) / (cmu25 * kappa_ * y);
                        const scalar wOmega =
                            Kokkos::min(Kokkos::sqrt(wVis * wVis + wLog * wLog), omegaMax);
                        Kokkos::atomic_add(&valuesV[owner], cw * wOmega);
                        maskV[owner] = scalar(1);
                    },
                    "pin_omega_wall_cells::accumulate"
                );
            }
            omegaPde.setConstraintsOwned(mask, values);
        },
        "omega_pde"_a,
        "omega"_a,
        "k"_a,
        "nu"_a,
        "near_wall_dist"_a,
        "runtime"_a,
        "beta1"_a = 0.075,
        "cmu"_a = 0.09,
        "kappa"_a = 0.41,
        "Pin the omega equation's near-wall cells to the omegaWallFunction blended value"
    );
}

} // namespace NeoFOAM::bindings
