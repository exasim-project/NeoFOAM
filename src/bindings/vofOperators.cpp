// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <string>

#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include "NeoFOAM/algorithms/vofOperators.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "fvCFD.H"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

void registerVofOperators(nb::module_& m)
{
    // Upwind convective flux phi_f * alpha_upwind(f) — the MULESCorr implicit-predictor
    // base flux (alpha1Eqn.flux() for an upwind div). Boundary = phi_b * alpha_b.
    m.def(
        "upwind_flux",
        [](const fvcc::VolumeField<NeoN::scalar>& alpha,
           const fvcc::SurfaceField<NeoN::scalar>& phi) -> fvcc::SurfaceField<NeoN::scalar>
        {
            const auto exec = alpha.exec();
            const auto& mesh = alpha.mesh();
            fvcc::SurfaceField<NeoN::scalar> flux(
                exec,
                "alphaPhiUpwind",
                mesh,
                fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh)
            );
            const auto own = mesh.faceOwners().view();
            const auto nei = mesh.faceNeighbors().view();
            const auto aI = alpha.internalVector().view();
            const auto phiI = phi.internalVector().view();
            auto fI = flux.internalVector().view();
            NeoN::parallelFor(
                exec,
                {0, mesh.nInternalFaces()},
                KOKKOS_LAMBDA(const NeoN::localIdx f) {
                    fI[f] = phiI[f] * (phiI[f] >= 0.0 ? aI[own[f]] : aI[nei[f]]);
                },
                "vof::upwindFluxInternal"
            );
            const auto aB = alpha.boundaryData().value().view();
            const auto phiB = phi.boundaryData().value().view();
            auto fB = flux.boundaryData().value().view();
            NeoN::parallelFor(
                exec,
                {0, mesh.nBoundaryFaces()},
                KOKKOS_LAMBDA(const NeoN::localIdx bf) { fB[bf] = phiB[bf] * aB[bf]; },
                "vof::upwindFluxBoundary"
            );
            return flux;
        },
        "alpha"_a,
        "phi"_a,
        "Upwind convective flux phi_f * alpha_upwind(f)"
    );

    // MULES::explicitSolve — bounded explicit FCT alpha advance (explicit path).
    m.def(
        "mules_explicit_solve",
        [](fvcc::VolumeField<NeoN::scalar>& alpha,
           const fvcc::SurfaceField<NeoN::scalar>& phi,
           fvcc::SurfaceField<NeoN::scalar>& alphaPhi,
           double deltaT,
           double psiMax,
           double psiMin,
           int nLimiterIter)
        { nf::mulesExplicitSolve(alpha, phi, alphaPhi, deltaT, psiMax, psiMin, nLimiterIter); },
        "alpha"_a,
        "phi"_a,
        "alpha_phi"_a,
        "delta_t"_a,
        "psi_max"_a = 1.0,
        "psi_min"_a = 0.0,
        "n_limiter_iter"_a = 3,
        "Bounded explicit MULES (FCT) solve (MULES::explicitSolve, simplified VoF path: "
        "rho=1, Sp=Su=0, static mesh). Limits alpha_phi in place and advances alpha "
        "conservatively (no clamp)."
    );

    // MULES::correct — FCT-limit + apply the antidiffusive correction flux (MULESCorr).
    m.def(
        "mules_correct",
        [](fvcc::VolumeField<NeoN::scalar>& alpha,
           fvcc::SurfaceField<NeoN::scalar>& phiCorr,
           double deltaT,
           int nLimiterIter,
           double relaxOld) { nf::mulesCorrect(alpha, phiCorr, deltaT, nLimiterIter, relaxOld); },
        "alpha"_a,
        "phiCorr"_a,
        "deltaT"_a,
        "nLimiterIter"_a = 3,
        "relaxOld"_a = 0.0,
        "MULES::correct: FCT-limit the correction flux and apply it to alpha "
        "(relaxOld>0 under-relaxes vs the pre-correction field, interFoam aCorr>0)"
    );

    // interFoam alphaEqn.H high-order phase flux (vanLeer + cAlpha interface compression).
    m.def(
        "alpha_phase_flux",
        [](nf::RunTime& rt,
           const fvcc::VolumeField<NeoN::scalar>& alpha1,
           const fvcc::SurfaceField<NeoN::scalar>& phi,
           double cAlpha) -> fvcc::SurfaceField<NeoN::scalar>
        { return nf::alphaPhaseFlux(rt, alpha1, phi, cAlpha); },
        "runtime"_a,
        "alpha1"_a,
        "phi"_a,
        "cAlpha"_a,
        "interFoam alphaPhiUn = vanLeer flux + cAlpha interface compression"
    );

    // fvc::flux(phi, alpha, "Gauss vanLeer") — vanLeer-limited convective alpha flux.
    m.def(
        "vanleer_flux",
        [](nf::RunTime& rt,
           const fvcc::VolumeField<NeoN::scalar>& alpha,
           const fvcc::SurfaceField<NeoN::scalar>& phi) -> fvcc::SurfaceField<NeoN::scalar>
        { return nf::vanLeerAlphaFlux(rt, alpha, phi); },
        "runtime"_a,
        "alpha"_a,
        "phi"_a,
        "vanLeer-limited convective flux fvc::flux(phi, alpha, 'Gauss vanLeer')"
    );

    // snGrad(field): face-normal gradient (uncorrected scheme). Returns a fresh
    // SurfaceField<scalar> by value.
    m.def(
        "sn_grad",
        [](const fvcc::VolumeField<NeoN::scalar>& f) -> fvcc::SurfaceField<NeoN::scalar>
        {
            fvcc::FaceNormalGradient<NeoN::scalar> fng(
                f.exec(),
                f.mesh(),
                NeoN::Input {NeoN::TokenList {std::string("uncorrected")}}
            );
            return fng.faceNormalGrad(f);
        },
        "field"_a,
        "Face-normal gradient snGrad(field), uncorrected scheme. ORTHOGONAL MESHES ONLY: "
        "uncorrected == corrected on an orthogonal mesh; silently wrong on non-orthogonal meshes."
    );

    // Element-wise product of two scalar volume fields (internal + boundary), as a
    // fresh unregistered VolumeField. Used to form rho*rAU for interFoam's ddtCorr
    // weighting interpolate(rho*rAU) (NeoN has no VolumeField __mul__ binding).
    m.def(
        "mul_scalar_volume",
        [](const fvcc::VolumeField<NeoN::scalar>& a,
           const fvcc::VolumeField<NeoN::scalar>& b) -> fvcc::VolumeField<NeoN::scalar>
        {
            fvcc::VolumeField<NeoN::scalar> res(
                a.exec(),
                a.name + "*" + b.name,
                a.mesh(),
                fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(a.mesh())
            );
            res.internalVector() = a.internalVector();
            NeoN::mul(res.internalVector(), b.internalVector());
            res.boundaryData().value() = a.boundaryData().value();
            NeoN::mul(res.boundaryData().value(), b.boundaryData().value());
            return res;
        },
        "a"_a,
        "b"_a,
        "Element-wise product of two scalar volume fields (internal + boundary)"
    );

    // |Sf|: face-area magnitudes as a surface field, built from the OpenFOAM mesh's
    // magSf() and converted (unregistered) — mirrors the ghf/rhoPhi conversion path.
    m.def(
        "mag_sf",
        [](nf::RunTime& rt) -> fvcc::SurfaceField<NeoN::scalar>
        {
            Foam::surfaceScalarField magSf("magSf", rt.mesh.magSf());
            return nf::constructFrom(rt.exec, rt.nfMesh, magSf);
        },
        "runtime"_a,
        "Face-area magnitudes |Sf| as a surface field"
    );

    // Interface unit-normal flux nHatf = nHatfv & Sf (interfaceProperties::calculateK).
    m.def(
        "interface_nhatf",
        [](nf::RunTime&, const fvcc::VolumeField<NeoN::scalar>& alpha1
        ) -> fvcc::SurfaceField<NeoN::scalar> { return nf::computeNHatf(alpha1); },
        "runtime"_a,
        "alpha1"_a,
        "Interface unit-normal flux nHatf = nHatfv & Sf (interfaceProperties::calculateK)"
    );

    // Surface-tension face force fSigma = interpolate(sigma*K)*snGrad(alpha1), with the
    // curvature K = -div(nHatf) = -surfaceIntegrate(nHatf). Mirrors
    // interfaceProperties::surfaceTensionForce (interfaceProperties.C:237-241).
    m.def(
        "surface_tension_force",
        [](nf::RunTime&, const fvcc::VolumeField<NeoN::scalar>& alpha1, double sigma
        ) -> fvcc::SurfaceField<NeoN::scalar> { return nf::surfaceTensionForce(alpha1, sigma); },
        "runtime"_a,
        "alpha1"_a,
        "sigma"_a,
        "Surface-tension face force sigma*K*snGrad(alpha1) (interfaceProperties). "
        "ORTHOGONAL MESHES ONLY: uses the uncorrected snGrad scheme (uncorrected == corrected "
        "on an orthogonal mesh); silently wrong on non-orthogonal meshes."
    );
}
} // namespace NeoFOAM::bindings
