// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Turbulence-model bindings for the Python NeoN RAS/LES closures:
//   - the runtime-selected TurbulenceModel handle (owns nut/nuTilda/gradU),
//   - the Gauss-Green velocity-gradient tensor and the explicit dev2 viscous
//     stress that consume it,
//   - the scalar strain/vorticity invariants the two-equation and Spalart-
//     Allmaras production terms are built from (small tensor/vector
//     contractions kept in Kokkos kernels, no field-level tensor algebra in
//     the Python DSL).

#include <nanobind/nanobind.h>

// NeoN headers
#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp" // GaussGreenGrad

// NeoFOAM headers
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoFOAM/fvcc/operators/viscousStressOperator.hpp"
#include "NeoFOAM/turbulenceModels/turbulenceModel.hpp"

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

// Foam::bound(volScalarField&, const dimensionedScalar&), src/finiteVolume/cfdTools/
// general/bound/bound.C. A cell at or below the floor is NOT clipped to the floor: it
// takes fvc::average(max(field, lower)) — the face-area-weighted average of the floored
// neighbourhood — so a bounded epsilon/omega keeps the scale of its neighbours. Clipping
// to the floor instead would leave nut = Cmu k^2/epsilon dividing by ~1e-15.
//
// OpenFOAM guards the whole update on a *global* min < lower. Above the floor the update
// is the identity, so the guard only skips redundant work: taking the local min here is
// equivalent cell-for-cell and needs no MPI reduction.
void boundLower(fvcc::VolumeField<NeoN::scalar>& field, NeoN::scalar lowerBound)
{
    using NeoN::localIdx;
    using NeoN::scalar;

    const auto& mesh = field.mesh();
    const auto exec = field.exec();
    const auto nCells = mesh.nCells();

    scalar minValue = 0;
    {
        const auto fieldV = field.internalVector().view();
        Kokkos::Min<scalar> reduceMin(minValue);
        NeoN::parallelReduce(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx i, scalar& s) {
                if (fieldV[i] < s) s = fieldV[i];
            },
            reduceMin
        );
    }
    if (minValue >= lowerBound) return;

    NeoN::Logging::info("bounding {}, min: {}", field.name, minValue);

    // floored = max(field, lower), internal and boundary — the field fvc::average sees.
    fvcc::VolumeField<scalar> floored(
        exec,
        "bound(" + field.name + ")",
        mesh,
        fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh)
    );
    {
        const auto fieldV = field.internalVector().view();
        auto flooredV = floored.internalVector().view();
        NeoN::parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx i) { flooredV[i] = Kokkos::max(fieldV[i], lowerBound); },
            "bound::floorInternal"
        );
        const auto fieldB = field.boundaryData().value().view();
        auto flooredB = floored.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, flooredB.size()},
            NEON_LAMBDA(const localIdx i) { flooredB[i] = Kokkos::max(fieldB[i], lowerBound); },
            "bound::floorBoundary"
        );
    }

    const fvcc::SurfaceInterpolation<scalar> surfInterp(
        exec,
        mesh,
        NeoN::TokenList({std::string("linear")})
    );
    const auto surfFloored = surfInterp.interpolate(floored);

    // fvc::average(ssf) = sum_f |Sf| ssf_f / sum_f |Sf| over each cell's faces.
    NeoN::Vector<scalar> numerator(exec, nCells, scalar(0));
    NeoN::Vector<scalar> denominator(exec, nCells, scalar(0));
    {
        const auto magSf = mesh.faceAreas().view();
        const auto owners = mesh.faceOwners().view();
        const auto neighbours = mesh.faceNeighbors().view();
        const auto boundaryOwners = mesh.boundaryMesh().faceOwners().view();
        const auto boundaryMagSf = mesh.boundaryMesh().faceAreas().view();
        const auto faceV = surfFloored.internalVector().view();
        const auto faceB = surfFloored.boundaryData().value().view();
        auto num = numerator.view();
        auto den = denominator.view();
        NeoN::parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                const scalar area = magSf[f];
                const scalar contribution = area * faceV[f];
                Kokkos::atomic_add(&num[owners[f]], contribution);
                Kokkos::atomic_add(&num[neighbours[f]], contribution);
                Kokkos::atomic_add(&den[owners[f]], area);
                Kokkos::atomic_add(&den[neighbours[f]], area);
            },
            "bound::averageInternalFaces"
        );
        NeoN::parallelFor(
            exec,
            {0, static_cast<localIdx>(boundaryOwners.size())},
            NEON_LAMBDA(const localIdx bf) {
                const scalar area = boundaryMagSf[bf];
                Kokkos::atomic_add(&num[boundaryOwners[bf]], area * faceB[bf]);
                Kokkos::atomic_add(&den[boundaryOwners[bf]], area);
            },
            "bound::averageBoundaryFaces"
        );
    }

    {
        auto fieldV = field.internalVector().view();
        const auto num = numerator.view();
        const auto den = denominator.view();
        NeoN::parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx i) {
                // max(max(field, average*pos0(-field)), lower): pos0(-x) is 1 for x <= 0.
                const scalar repaired = (fieldV[i] <= scalar(0)) ? (num[i] / den[i]) : fieldV[i];
                fieldV[i] = Kokkos::max(repaired, lowerBound);
            },
            "bound::repair"
        );
        auto fieldB = field.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, fieldB.size()},
            NEON_LAMBDA(const localIdx i) { fieldB[i] = Kokkos::max(fieldB[i], lowerBound); },
            "bound::boundary"
        );
    }
}
} // namespace

namespace NeoFOAM::bindings
{

void registerTurbulenceModel(nb::module_& m)
{
    // -------------------------------------------------------------------
    // VolumeField<Tensor> — opaque handle so Python can hold the velocity
    // gradient between gradTensor() and the viscousStress() operator.
    // -------------------------------------------------------------------
    nb::class_<fvcc::VolumeField<NeoN::Tensor>>(m, "TensorVolumeField");

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
    // Spalart-Allmaras / kOmegaSST primitives that have no field-level counterpart
    // in the Python DSL: the vorticity magnitude and the scalar-gradient magnitude
    // squared (both small tensor/vector contractions kept in Kokkos kernels).
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
        "bound",
        &boundLower,
        "field"_a,
        "lower"_a,
        "Foam::bound: sub-floor cells take fvc::average(max(field, lower)), not the floor"
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
}

} // namespace NeoFOAM::bindings
