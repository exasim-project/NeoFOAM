// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>
#include <nanobind/nanobind.h>
#include <string>

#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
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

namespace
{

// Interface unit-normal flux nHatf = nHatfv & Sf, with
//   nHatfv     = gradAlphaf / (|gradAlphaf| + deltaN),
//   gradAlphaf = interpolate(grad(alpha1))   (Gauss-Green grad, linear face interp),
//   deltaN     = 1e-8 / cbrt(mean(cell volume)).
// Faithful to interfaceProperties::calculateK (interfaceProperties.C:108-166) minus the
// contact-angle correction and curvature smoothing. Scatters internal + boundary faces via
// Kokkos parallelFor over the mesh Sf views (reconstruct.cpp pattern). Factored out so
// surface_tension_force reuses the identical nHatf (single source of truth).
fvcc::SurfaceField<NeoN::scalar> computeNHatf(const fvcc::VolumeField<NeoN::scalar>& alpha1)
{
    const auto exec = alpha1.exec();
    const auto& mesh = alpha1.mesh();

    // Cell gradient of alpha (Gauss-Green == OpenFOAM "nHat" Gauss linear default).
    fvcc::GaussGreenGrad gg(exec, mesh);
    fvcc::VolumeField<NeoN::Vec3> gradAlpha = gg.grad(alpha1);

    // Interpolated face gradient gradAlphaf = interpolate(gradAlpha) (linear).
    fvcc::SurfaceInterpolation<NeoN::Vec3> interp(
        exec, mesh, NeoN::Input {NeoN::TokenList {std::string("linear")}}
    );
    fvcc::SurfaceField<NeoN::Vec3> gradAlphaf = interp.interpolate(gradAlpha);

    // deltaN = 1e-8 / cbrt(mean(cell volume)); must match interfaceProperties.C:194 exactly.
    const auto nCells = mesh.nCells();
    auto vHost = mesh.cellVolumes().copyToHost();
    auto vView = vHost.view();
    double sumV = 0.0;
    for (NeoN::localIdx c = 0; c < nCells; ++c) sumV += vView[c];
    const NeoN::scalar deltaN = 1e-8 / std::cbrt(sumV / static_cast<double>(nCells));

    fvcc::SurfaceField<NeoN::scalar> nHatf(
        exec, "nHatf", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh)
    );

    const auto Sf = mesh.faceNormals().view();
    const auto gaf = gradAlphaf.internalVector().view();
    auto nh = nHatf.internalVector().view();
    NeoN::parallelFor(
        exec, {0, mesh.nInternalFaces()},
        KOKKOS_LAMBDA(const NeoN::localIdx f) {
            const NeoN::Vec3 g = gaf[f];
            const NeoN::scalar mg = NeoN::mag(g) + deltaN;
            nh[f] = (g / mg) & Sf[f];
        },
        "vof::nHatfInternal"
    );

    const auto bSf = mesh.boundaryMesh().faceNormals().view();
    const auto bgaf = gradAlphaf.boundaryData().value().view();
    auto bnh = nHatf.boundaryData().value().view();
    NeoN::parallelFor(
        exec, {0, mesh.nBoundaryFaces()},
        KOKKOS_LAMBDA(const NeoN::localIdx bf) {
            const NeoN::Vec3 g = bgaf[bf];
            const NeoN::scalar mg = NeoN::mag(g) + deltaN;
            bnh[bf] = (g / mg) & bSf[bf];
        },
        "vof::nHatfBoundary"
    );

    return nHatf;
}

// fvc::flux(phi, alpha, "Gauss vanLeer"): the vanLeer-limited convective face flux
// phi_f * alpha_f, where alpha_f uses the limited interpolation weight
//   w_f       = limiter*wCD + (1 - limiter)*pos0(phi_f)          (limitedSurfaceInterpolationScheme)
//   wCD       = SfdNei/(SfdOwn + SfdNei)                          (linear/CD weight, makeWeights)
//   limiter   = (r + |r|)/(1 + |r|)                              (vanLeerLimiter)
//   r         = NVDTVD::r(phi_f, alpha_own, alpha_nei, gradc_own, gradc_nei, d)  (see below)
//   gradc     = fvc::grad(alpha)  (Gauss linear == GaussGreenGrad),  d = C_nei - C_own
// Faithful transcription of Foam::LimitedScheme::calcLimiter + NVDTVD::r + vanLeerLimiter
// (src/finiteVolume/interpolation/surfaceInterpolation/limited*). Boundary flux is
// phi_b * alpha_b (non-coupled patch value), matching fvc::flux on physical patches.
fvcc::SurfaceField<NeoN::scalar> vanLeerAlphaFlux(
    nf::RunTime& rt,
    const fvcc::VolumeField<NeoN::scalar>& alpha,
    const fvcc::SurfaceField<NeoN::scalar>& phi
)
{
    const auto exec = alpha.exec();
    const auto& mesh = alpha.mesh();

    // Cell gradient of alpha (Gauss-Green == OpenFOAM fvc::grad(alpha) Gauss linear default).
    fvcc::GaussGreenGrad gg(exec, mesh);
    fvcc::VolumeField<NeoN::Vec3> gradAlpha = gg.grad(alpha);

    // Linear CD weights and cell-to-cell delta d = C_nei - C_own, sourced from the OpenFOAM
    // mesh (the NeoN adapter mesh does not populate cellCenters/faceCenters, so they cannot be
    // used here; taking OpenFOAM's weights()/C() also makes the CD weight bit-exact). d on
    // boundary faces is unused (boundary flux is phi_b*alpha_b).
    auto wN = nf::constructFrom(rt.exec, rt.nfMesh, rt.mesh.weights());
    Foam::surfaceVectorField dField(
        Foam::IOobject(
            "vanLeerDelta", rt.mesh.time().timeName(), rt.mesh,
            Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE, false
        ),
        rt.mesh,
        Foam::dimensionedVector("d", Foam::dimLength, Foam::vector::zero)
    );
    {
        Foam::vectorField& dIf = dField.primitiveFieldRef();
        const Foam::vectorField& CC = rt.mesh.C();
        const Foam::labelUList& fOwn = rt.mesh.owner();
        const Foam::labelUList& fNei = rt.mesh.neighbour();
        forAll(dIf, f) dIf[f] = CC[fNei[f]] - CC[fOwn[f]];
    }
    auto dN = nf::constructFrom(rt.exec, rt.nfMesh, dField);

    fvcc::SurfaceField<NeoN::scalar> flux(
        exec, "alphaPhiVanLeer", mesh,
        fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh)
    );

    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto aI = alpha.internalVector().view();
    const auto gI = gradAlpha.internalVector().view();
    const auto phiI = phi.internalVector().view();
    const auto wCDv = wN.internalVector().view();
    const auto dv = dN.internalVector().view();
    auto fI = flux.internalVector().view();

    NeoN::parallelFor(
        exec, {0, mesh.nInternalFaces()},
        KOKKOS_LAMBDA(const NeoN::localIdx f) {
            const NeoN::localIdx o = own[f];
            const NeoN::localIdx n = nei[f];
            const NeoN::scalar wCD = wCDv[f];
            const NeoN::Vec3 d = dv[f];
            const NeoN::scalar phif = phiI[f];
            const NeoN::scalar gradf = aI[n] - aI[o];
            const NeoN::scalar gradcf = (phif > 0.0) ? (d & gI[o]) : (d & gI[n]);

            NeoN::scalar r;
            if (Kokkos::fabs(gradcf) >= 1000.0 * Kokkos::fabs(gradf))
            {
                const NeoN::scalar sgc =
                    (gradcf > 0.0) ? 1.0 : ((gradcf < 0.0) ? -1.0 : 0.0);
                const NeoN::scalar sgf =
                    (gradf > 0.0) ? 1.0 : ((gradf < 0.0) ? -1.0 : 0.0);
                r = 2.0 * 1000.0 * sgc * sgf - 1.0;
            }
            else
            {
                r = 2.0 * (gradcf / gradf) - 1.0;
            }
            const NeoN::scalar limiter = (r + Kokkos::fabs(r)) / (1.0 + Kokkos::fabs(r));

            const NeoN::scalar wUp = (phif >= 0.0) ? NeoN::scalar(1) : NeoN::scalar(0);
            const NeoN::scalar w = limiter * wCD + (1.0 - limiter) * wUp;
            const NeoN::scalar alphaf = w * aI[o] + (1.0 - w) * aI[n];
            fI[f] = phif * alphaf;
        },
        "vof::vanLeerFluxInternal"
    );

    // Boundary flux = phi_b * alpha_b (non-coupled patch value).
    const auto aB = alpha.boundaryData().value().view();
    const auto phiB = phi.boundaryData().value().view();
    auto fB = flux.boundaryData().value().view();
    NeoN::parallelFor(
        exec, {0, mesh.nBoundaryFaces()},
        KOKKOS_LAMBDA(const NeoN::localIdx bf) { fB[bf] = phiB[bf] * aB[bf]; },
        "vof::vanLeerFluxBoundary"
    );

    return flux;
}

// interFoam alphaEqn.H high-order phase flux alphaPhiUn:
//   alphaPhiUn = fvc::flux(phi, alpha1, "div(phi,alpha)")                       [vanLeer]
//              + fvc::flux(-fvc::flux(-phir, alpha2, "div(phirb,alpha)"),
//                          alpha1, "div(phirb,alpha)")                          [compression]
// with phir = phic*nHatf, phic = cAlpha*|phi/magSf| (non-coupled boundary phic == 0),
// alphaScheme = Gauss vanLeer, alpharScheme = Gauss linear. Both alpharScheme interpolations
// are linear, so the compression term reduces to phir*linearInterp(alpha1)*linearInterp(alpha2);
// with alpha2 = 1 - alpha1 and linearInterp affine, linearInterp(alpha2) = 1 - linearInterp(alpha1).
// Faithful to VoF/alphaEqn.H (icAlpha == scAlpha == 0, the damBreak / interFoam default).
fvcc::SurfaceField<NeoN::scalar> alphaPhaseFlux(
    nf::RunTime& rt,
    const fvcc::VolumeField<NeoN::scalar>& alpha1,
    const fvcc::SurfaceField<NeoN::scalar>& phi,
    double cAlpha
)
{
    fvcc::SurfaceField<NeoN::scalar> flux = vanLeerAlphaFlux(rt, alpha1, phi);
    if (cAlpha == 0.0)
    {
        return flux;
    }

    const auto exec = alpha1.exec();
    const auto& mesh = alpha1.mesh();

    fvcc::SurfaceField<NeoN::scalar> nHatf = computeNHatf(alpha1);
    fvcc::SurfaceInterpolation<NeoN::scalar> interp(
        exec, mesh, NeoN::Input {NeoN::TokenList {std::string("linear")}}
    );
    fvcc::SurfaceField<NeoN::scalar> a1f = interp.interpolate(alpha1);

    const auto phiI = phi.internalVector().view();
    const auto magSf = mesh.faceAreas().view();  // |Sf| per internal face (populated)
    const auto nh = nHatf.internalVector().view();
    const auto a1fI = a1f.internalVector().view();
    auto fI = flux.internalVector().view();

    NeoN::parallelFor(
        exec, {0, mesh.nInternalFaces()},
        KOKKOS_LAMBDA(const NeoN::localIdx f) {
            const NeoN::scalar phic = cAlpha * Kokkos::fabs(phiI[f] / magSf[f]);
            const NeoN::scalar phir = phic * nh[f];
            fI[f] += phir * a1fI[f] * (1.0 - a1fI[f]);
        },
        "vof::alphaPhaseFluxCompression"
    );

    return flux;
}

// Bounded explicit MULES (FCT) solve — Foam::MULES::explicitSolve / limit / limiter
// (OpenFOAM MULESTemplates.C) for the simplified VoF path: rho == 1, Sp == Su == 0,
// static mesh, extremaCoeff == smoothLimiter == 0, no coupled/fixed-value boundary
// widening. Limits @p alphaPhi in place (high-order flux in, FCT-limited out) and
// advances @p alpha conservatively (no clamp — boundedness comes from the limiter).
// @p alpha is the current == oldTime field (psi0) at entry. This is a NeoFOAM-level
// VoF algorithm composed from NeoN building blocks (parallelFor / surfaceIntegrate /
// mesh views / Vector), moved here so NeoN stays algorithm-free; its sibling
// mulesCorrect lives just below.
void mulesExplicitSolve(
    fvcc::VolumeField<NeoN::scalar>& alpha,
    const fvcc::SurfaceField<NeoN::scalar>& phi,
    fvcc::SurfaceField<NeoN::scalar>& alphaPhi,
    NeoN::scalar deltaT,
    NeoN::scalar psiMax,
    NeoN::scalar psiMin,
    int nLimiterIter
)
{
    using NeoN::localIdx;
    using NeoN::scalar;
    const auto& mesh = alpha.mesh();
    const auto exec = alpha.exec();
    const auto nCells = mesh.nCells();
    const auto nInt = mesh.nInternalFaces();
    const auto nBnd = mesh.nBoundaryFaces();
    const scalar rDeltaT = 1.0 / deltaT;
    const scalar ROOTVSMALL = 1e-18;

    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto V = mesh.cellVolumes().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();

    auto a = alpha.internalVector().view();     // current == oldTime (psi0) at entry
    auto ap = alphaPhi.internalVector().view(); // high-order flux in; limited out
    const auto phiI = phi.internalVector().view();
    const auto apB = alphaPhi.boundaryData().value().view();

    // donor (upwind) flux phiBD + correction phiCorr on internal faces
    // (boundary: phiBD == alphaPhi ⇒ phiCorr == 0).
    NeoN::Vector<scalar> PhiBD(exec, nInt, 0.0), PhiCorr(exec, nInt, 0.0);
    auto phiBD = PhiBD.view();
    auto phiCorr = PhiCorr.view();
    NeoN::parallelFor(
        exec, {0, nInt},
        KOKKOS_LAMBDA(const localIdx f) {
            const scalar donor = (phiI[f] >= 0.0) ? a[own[f]] : a[nei[f]];
            phiBD[f] = donor * phiI[f];
            phiCorr[f] = ap[f] - phiBD[f];
        },
        "mules::phiBD"
    );

    // limiter: per-cell allowable extrema and bound RHS.
    NeoN::Vector<scalar> PsiMaxn(exec, nCells, psiMin), PsiMinn(exec, nCells, psiMax);
    NeoN::Vector<scalar> SumPhiBD(exec, nCells, 0.0), SumPhip(exec, nCells, 0.0),
        MSumPhim(exec, nCells, 0.0);
    auto psiMaxn = PsiMaxn.view();
    auto psiMinn = PsiMinn.view();
    auto sumPhiBD = SumPhiBD.view();
    auto sumPhip = SumPhip.view();
    auto mSumPhim = MSumPhim.view();

    NeoN::parallelFor(
        exec, {0, nInt},
        KOKKOS_LAMBDA(const localIdx f) {
            const localIdx o = own[f];
            const localIdx n = nei[f];
            Kokkos::atomic_max(&psiMaxn[o], a[n]);
            Kokkos::atomic_min(&psiMinn[o], a[n]);
            Kokkos::atomic_max(&psiMaxn[n], a[o]);
            Kokkos::atomic_min(&psiMinn[n], a[o]);
            Kokkos::atomic_add(&sumPhiBD[o], phiBD[f]);
            Kokkos::atomic_sub(&sumPhiBD[n], phiBD[f]);
            const scalar pc = phiCorr[f];
            if (pc > 0.0)
            {
                Kokkos::atomic_add(&sumPhip[o], pc);
                Kokkos::atomic_add(&mSumPhim[n], pc);
            }
            else
            {
                Kokkos::atomic_sub(&mSumPhim[o], pc);
                Kokkos::atomic_sub(&sumPhip[n], pc);
            }
        },
        "mules::extremaInternal"
    );
    NeoN::parallelFor(
        exec, {0, nBnd},
        KOKKOS_LAMBDA(const localIdx bf) { Kokkos::atomic_add(&sumPhiBD[bOwn[bf]], apB[bf]); },
        "mules::extremaBoundary"
    );
    NeoN::parallelFor(
        exec, {0, nCells},
        KOKKOS_LAMBDA(const localIdx c) {
            const scalar mx = Kokkos::min(psiMaxn[c], psiMax);
            const scalar mn = Kokkos::max(psiMinn[c], psiMin);
            const scalar vr = V[c] * rDeltaT;
            psiMaxn[c] = vr * (mx - a[c]) + sumPhiBD[c];
            psiMinn[c] = vr * (a[c] - mn) - sumPhiBD[c];
        },
        "mules::boundRHS"
    );

    // FCT limiter sweeps.
    NeoN::Vector<scalar> Lambda(exec, nInt, 1.0);
    auto lambda = Lambda.view();
    NeoN::Vector<scalar> SumlPhip(exec, nCells, 0.0), MSumlPhim(exec, nCells, 0.0);
    auto sumlPhip = SumlPhip.view();
    auto mSumlPhim = MSumlPhim.view();
    for (int j = 0; j < nLimiterIter; ++j)
    {
        NeoN::parallelFor(
            exec, {0, nCells},
            KOKKOS_LAMBDA(const localIdx c) { sumlPhip[c] = 0.0; mSumlPhim[c] = 0.0; },
            "mules::zeroSuml"
        );
        NeoN::parallelFor(
            exec, {0, nInt},
            KOKKOS_LAMBDA(const localIdx f) {
                const scalar lpc = lambda[f] * phiCorr[f];
                if (lpc > 0.0)
                {
                    Kokkos::atomic_add(&sumlPhip[own[f]], lpc);
                    Kokkos::atomic_add(&mSumlPhim[nei[f]], lpc);
                }
                else
                {
                    Kokkos::atomic_sub(&mSumlPhim[own[f]], lpc);
                    Kokkos::atomic_sub(&sumlPhip[nei[f]], lpc);
                }
            },
            "mules::sumlInternal"
        );
        NeoN::parallelFor(
            exec, {0, nCells},
            KOKKOS_LAMBDA(const localIdx c) {
                sumlPhip[c] = Kokkos::max(
                    Kokkos::min((sumlPhip[c] + psiMaxn[c]) / (mSumPhim[c] + ROOTVSMALL), 1.0), 0.0
                );
                mSumlPhim[c] = Kokkos::max(
                    Kokkos::min((mSumlPhim[c] + psiMinn[c]) / (sumPhip[c] + ROOTVSMALL), 1.0), 0.0
                );
            },
            "mules::lambdaCells"
        );
        auto lambdam = sumlPhip;
        auto lambdap = mSumlPhim;
        NeoN::parallelFor(
            exec, {0, nInt},
            KOKKOS_LAMBDA(const localIdx f) {
                if (phiCorr[f] > 0.0)
                    lambda[f] = Kokkos::min(lambda[f], Kokkos::min(lambdap[own[f]], lambdam[nei[f]]));
                else
                    lambda[f] = Kokkos::min(lambda[f], Kokkos::min(lambdam[own[f]], lambdap[nei[f]]));
            },
            "mules::lambdaFaces"
        );
    }

    // apply limiter: alphaPhi = phiBD + lambda*phiCorr (internal; boundary unchanged).
    NeoN::parallelFor(
        exec, {0, nInt},
        KOKKOS_LAMBDA(const localIdx f) { ap[f] = phiBD[f] + lambda[f] * phiCorr[f]; },
        "mules::applyLimiter"
    );

    // conservative explicit update: alpha = psi0 - deltaT*surfaceIntegrate(alphaPhi).
    NeoN::Vector<scalar> Div(exec, nCells, 0.0);
    fvcc::surfaceIntegrate<NeoN::scalar>(
        exec, nInt, nei, own, bOwn, alphaPhi.internalVector().view(), apB, V, Div.view(),
        NeoN::dsl::Coeff(1.0)
    );
    auto div = Div.view();
    NeoN::parallelFor(
        exec, {0, nCells},
        KOKKOS_LAMBDA(const localIdx c) { a[c] = a[c] - deltaT * div[c]; },
        "mules::update"
    );
    alpha.correctBoundaryConditions();
}


// MULES::correct (CMULESTemplates.C limiterCorr + correct) for the VoF path
// (rho==1, Sp==Su==0, extremaCoeff==smoothLimiter==0, psiMax==1, psiMin==0):
// FCT-limit the antidiffusive correction flux @p phiCorr so applying it keeps
// alpha bounded, then apply it — alpha := alpha - deltaT*surfaceIntegrate(phiCorr).
// @p alpha is the predictor-advanced field on entry; both @p alpha and @p phiCorr
// are modified in place (phiCorr becomes the limited correction). Same FCT sweep as
// mulesExplicitSolve's limiter minus the donor (phiBD) split — the base state is the
// current alpha, the bound RHS is V*rDeltaT*(psiMaxn - alpha).
// @p relaxOld under-relaxes the corrected field against its pre-correction state
// (interFoam's aCorr>0 branch: ``alpha1 = 0.5*alpha1 + 0.5*alpha10``): the field
// is snapshotted on entry and, after the limit+apply, blended
// ``alpha = (1 - relaxOld)*alpha_corrected + relaxOld*alpha_pre``. relaxOld = 0
// (the default) reproduces the first-corrector behaviour (no under-relaxation).
void mulesCorrect(
    fvcc::VolumeField<NeoN::scalar>& alpha,
    fvcc::SurfaceField<NeoN::scalar>& phiCorr,
    double deltaT,
    int nLimiterIter,
    double relaxOld
)
{
    using NeoN::localIdx;
    using NeoN::scalar;
    const auto& mesh = alpha.mesh();
    const auto exec = alpha.exec();
    const auto nCells = mesh.nCells();

    // Snapshot the pre-correction field for the optional under-relaxation.
    NeoN::Vector<scalar> AlphaOld(exec, nCells, 0.0);
    if (relaxOld != 0.0)
    {
        auto aOldW = AlphaOld.view();
        const auto aIn = alpha.internalVector().view();
        NeoN::parallelFor(
            exec, {0, nCells},
            KOKKOS_LAMBDA(const localIdx c) { aOldW[c] = aIn[c]; },
            "mulesCorr::snapshot"
        );
    }
    const auto nInt = mesh.nInternalFaces();
    const auto nBnd = mesh.nBoundaryFaces();
    const scalar rDeltaT = 1.0 / deltaT;
    const scalar ROOTVSMALL = 1e-18;
    const scalar psiMax = 1.0, psiMin = 0.0;

    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto V = mesh.cellVolumes().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    const auto a = alpha.internalVector().view();
    const auto pc = phiCorr.internalVector().view();
    const auto pcB = phiCorr.boundaryData().value().view();

    NeoN::Vector<scalar> PsiMaxn(exec, nCells, psiMin), PsiMinn(exec, nCells, psiMax);
    NeoN::Vector<scalar> SumPhip(exec, nCells, 0.0), MSumPhim(exec, nCells, 0.0);
    auto psiMaxn = PsiMaxn.view();
    auto psiMinn = PsiMinn.view();
    auto sumPhip = SumPhip.view();
    auto mSumPhim = MSumPhim.view();

    NeoN::parallelFor(
        exec, {0, nInt},
        KOKKOS_LAMBDA(const localIdx f) {
            const localIdx o = own[f];
            const localIdx n = nei[f];
            Kokkos::atomic_max(&psiMaxn[o], a[n]);
            Kokkos::atomic_min(&psiMinn[o], a[n]);
            Kokkos::atomic_max(&psiMaxn[n], a[o]);
            Kokkos::atomic_min(&psiMinn[n], a[o]);
            const scalar c = pc[f];
            if (c > 0.0)
            {
                Kokkos::atomic_add(&sumPhip[o], c);
                Kokkos::atomic_add(&mSumPhim[n], c);
            }
            else
            {
                Kokkos::atomic_sub(&mSumPhim[o], c);
                Kokkos::atomic_sub(&sumPhip[n], c);
            }
        },
        "mulesCorr::extrema"
    );
    NeoN::parallelFor(
        exec, {0, nBnd},
        KOKKOS_LAMBDA(const localIdx bf) {
            const scalar c = pcB[bf];
            if (c > 0.0) Kokkos::atomic_add(&sumPhip[bOwn[bf]], c);
            else Kokkos::atomic_sub(&mSumPhim[bOwn[bf]], c);
        },
        "mulesCorr::extremaBnd"
    );
    NeoN::parallelFor(
        exec, {0, nCells},
        KOKKOS_LAMBDA(const localIdx c) {
            const scalar mx = Kokkos::min(psiMaxn[c], psiMax);
            const scalar mn = Kokkos::max(psiMinn[c], psiMin);
            const scalar vr = V[c] * rDeltaT;
            psiMaxn[c] = vr * (mx - a[c]);
            psiMinn[c] = vr * (a[c] - mn);
        },
        "mulesCorr::boundRHS"
    );

    NeoN::Vector<scalar> Lambda(exec, nInt, 1.0);
    auto lambda = Lambda.view();
    NeoN::Vector<scalar> SumlPhip(exec, nCells, 0.0), MSumlPhim(exec, nCells, 0.0);
    auto sumlPhip = SumlPhip.view();
    auto mSumlPhim = MSumlPhim.view();
    for (int j = 0; j < nLimiterIter; ++j)
    {
        NeoN::parallelFor(
            exec, {0, nCells},
            KOKKOS_LAMBDA(const localIdx c) { sumlPhip[c] = 0.0; mSumlPhim[c] = 0.0; },
            "mulesCorr::zero"
        );
        NeoN::parallelFor(
            exec, {0, nInt},
            KOKKOS_LAMBDA(const localIdx f) {
                const scalar lpc = lambda[f] * pc[f];
                if (lpc > 0.0)
                {
                    Kokkos::atomic_add(&sumlPhip[own[f]], lpc);
                    Kokkos::atomic_add(&mSumlPhim[nei[f]], lpc);
                }
                else
                {
                    Kokkos::atomic_sub(&mSumlPhim[own[f]], lpc);
                    Kokkos::atomic_sub(&sumlPhip[nei[f]], lpc);
                }
            },
            "mulesCorr::suml"
        );
        NeoN::parallelFor(
            exec, {0, nCells},
            KOKKOS_LAMBDA(const localIdx c) {
                sumlPhip[c] = Kokkos::max(
                    Kokkos::min((sumlPhip[c] + psiMaxn[c]) / (mSumPhim[c] + ROOTVSMALL), 1.0), 0.0
                );
                mSumlPhim[c] = Kokkos::max(
                    Kokkos::min((mSumlPhim[c] + psiMinn[c]) / (sumPhip[c] + ROOTVSMALL), 1.0), 0.0
                );
            },
            "mulesCorr::lambdaCells"
        );
        auto lambdam = sumlPhip;
        auto lambdap = mSumlPhim;
        NeoN::parallelFor(
            exec, {0, nInt},
            KOKKOS_LAMBDA(const localIdx f) {
                if (pc[f] > 0.0)
                    lambda[f] = Kokkos::min(lambda[f], Kokkos::min(lambdap[own[f]], lambdam[nei[f]]));
                else
                    lambda[f] = Kokkos::min(lambda[f], Kokkos::min(lambdam[own[f]], lambdap[nei[f]]));
            },
            "mulesCorr::lambdaFaces"
        );
    }

    // phiCorr *= lambda (internal; non-coupled boundary lambda stays 1).
    auto pcW = phiCorr.internalVector().view();
    NeoN::parallelFor(
        exec, {0, nInt},
        KOKKOS_LAMBDA(const localIdx f) { pcW[f] *= lambda[f]; },
        "mulesCorr::applyLambda"
    );

    // alpha := alpha - deltaT*surfaceIntegrate(phiCorr).
    NeoN::Vector<scalar> Div(exec, nCells, 0.0);
    fvcc::surfaceIntegrate<NeoN::scalar>(
        exec, nInt, nei, own, bOwn, phiCorr.internalVector().view(),
        phiCorr.boundaryData().value().view(), V, Div.view(), NeoN::dsl::Coeff(1.0)
    );
    auto div = Div.view();
    auto aW = alpha.internalVector().view();
    NeoN::parallelFor(
        exec, {0, nCells},
        KOKKOS_LAMBDA(const localIdx c) { aW[c] = aW[c] - deltaT * div[c]; },
        "mulesCorr::apply"
    );

    // Optional under-relaxation against the pre-correction snapshot.
    if (relaxOld != 0.0)
    {
        const auto aOld = AlphaOld.view();
        const scalar w = relaxOld;
        NeoN::parallelFor(
            exec, {0, nCells},
            KOKKOS_LAMBDA(const localIdx c) { aW[c] = (1.0 - w) * aW[c] + w * aOld[c]; },
            "mulesCorr::underRelax"
        );
    }

    alpha.correctBoundaryConditions();
}

} // namespace

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
                exec, "alphaPhiUpwind", mesh,
                fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh)
            );
            const auto own = mesh.faceOwners().view();
            const auto nei = mesh.faceNeighbors().view();
            const auto aI = alpha.internalVector().view();
            const auto phiI = phi.internalVector().view();
            auto fI = flux.internalVector().view();
            NeoN::parallelFor(
                exec, {0, mesh.nInternalFaces()},
                KOKKOS_LAMBDA(const NeoN::localIdx f) {
                    fI[f] = phiI[f] * (phiI[f] >= 0.0 ? aI[own[f]] : aI[nei[f]]);
                },
                "vof::upwindFluxInternal"
            );
            const auto aB = alpha.boundaryData().value().view();
            const auto phiB = phi.boundaryData().value().view();
            auto fB = flux.boundaryData().value().view();
            NeoN::parallelFor(
                exec, {0, mesh.nBoundaryFaces()},
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
        [](fvcc::VolumeField<NeoN::scalar>& alpha, const fvcc::SurfaceField<NeoN::scalar>& phi,
           fvcc::SurfaceField<NeoN::scalar>& alphaPhi, double deltaT, double psiMax,
           double psiMin, int nLimiterIter)
        { mulesExplicitSolve(alpha, phi, alphaPhi, deltaT, psiMax, psiMin, nLimiterIter); },
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
        [](fvcc::VolumeField<NeoN::scalar>& alpha, fvcc::SurfaceField<NeoN::scalar>& phiCorr,
           double deltaT, int nLimiterIter, double relaxOld)
        { mulesCorrect(alpha, phiCorr, deltaT, nLimiterIter, relaxOld); },
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
        [](nf::RunTime& rt, const fvcc::VolumeField<NeoN::scalar>& alpha1,
           const fvcc::SurfaceField<NeoN::scalar>& phi,
           double cAlpha) -> fvcc::SurfaceField<NeoN::scalar>
        { return alphaPhaseFlux(rt, alpha1, phi, cAlpha); },
        "runtime"_a,
        "alpha1"_a,
        "phi"_a,
        "cAlpha"_a,
        "interFoam alphaPhiUn = vanLeer flux + cAlpha interface compression"
    );

    // fvc::flux(phi, alpha, "Gauss vanLeer") — vanLeer-limited convective alpha flux.
    m.def(
        "vanleer_flux",
        [](nf::RunTime& rt, const fvcc::VolumeField<NeoN::scalar>& alpha,
           const fvcc::SurfaceField<NeoN::scalar>& phi) -> fvcc::SurfaceField<NeoN::scalar>
        { return vanLeerAlphaFlux(rt, alpha, phi); },
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
                f.exec(), f.mesh(),
                NeoN::Input {NeoN::TokenList {std::string("uncorrected")}}
            );
            return fng.faceNormalGrad(f);
        },
        "field"_a,
        "Face-normal gradient snGrad(field) (uncorrected)"
    );

    // Element-wise product of two scalar volume fields (internal + boundary), as a
    // fresh unregistered VolumeField. Used to form rho*rAU for interFoam's ddtCorr
    // weighting interpolate(rho*rAU) (NeoN has no VolumeField __mul__ binding).
    m.def(
        "mul_scalar_volume",
        [](const fvcc::VolumeField<NeoN::scalar>& a, const fvcc::VolumeField<NeoN::scalar>& b)
            -> fvcc::VolumeField<NeoN::scalar>
        {
            fvcc::VolumeField<NeoN::scalar> res(
                a.exec(), a.name + "*" + b.name, a.mesh(),
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
        [](nf::RunTime&, const fvcc::VolumeField<NeoN::scalar>& alpha1)
            -> fvcc::SurfaceField<NeoN::scalar> { return computeNHatf(alpha1); },
        "runtime"_a,
        "alpha1"_a,
        "Interface unit-normal flux nHatf = nHatfv & Sf (interfaceProperties::calculateK)"
    );

    // Surface-tension face force fSigma = interpolate(sigma*K)*snGrad(alpha1), with the
    // curvature K = -div(nHatf) = -surfaceIntegrate(nHatf). Mirrors
    // interfaceProperties::surfaceTensionForce (interfaceProperties.C:237-241).
    m.def(
        "surface_tension_force",
        [](nf::RunTime&, const fvcc::VolumeField<NeoN::scalar>& alpha1, double sigma)
            -> fvcc::SurfaceField<NeoN::scalar>
        {
            const auto exec = alpha1.exec();
            const auto& mesh = alpha1.mesh();

            fvcc::SurfaceField<NeoN::scalar> nHatf = computeNHatf(alpha1);

            // sigmaK = sigma*K = -sigma*surfaceIntegrate(nHatf). surfaceIntegrate divides by
            // V and scales by the operator coefficient, so passing -sigma yields sigma*K in
            // one pass (verified against reconstruct.cpp / mules.cpp surfaceIntegrate usage).
            fvcc::VolumeField<NeoN::scalar> sigmaK(
                exec, "sigmaK", mesh,
                fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh)
            );
            NeoN::fill(sigmaK.internalVector(), 0.0);
            fvcc::surfaceIntegrate<NeoN::scalar>(
                exec,
                mesh.nInternalFaces(),
                mesh.faceNeighbors().view(),
                mesh.faceOwners().view(),
                mesh.boundaryMesh().faceOwners().view(),
                nHatf.internalVector().view(),
                nHatf.boundaryData().value().view(),
                mesh.cellVolumes().view(),
                sigmaK.internalVector().view(),
                NeoN::dsl::Coeff(-sigma)
            );
            sigmaK.correctBoundaryConditions();

            // sigmaKf = interpolate(sigmaK) (linear).
            fvcc::SurfaceInterpolation<NeoN::scalar> interpS(
                exec, mesh, NeoN::Input {NeoN::TokenList {std::string("linear")}}
            );
            fvcc::SurfaceField<NeoN::scalar> sigmaKf = interpS.interpolate(sigmaK);

            // snGrad(alpha1) (uncorrected == corrected on the orthogonal damBreak mesh).
            fvcc::FaceNormalGradient<NeoN::scalar> fng(
                exec, mesh, NeoN::Input {NeoN::TokenList {std::string("uncorrected")}}
            );
            fvcc::SurfaceField<NeoN::scalar> snAlpha = fng.faceNormalGrad(alpha1);

            // fSigma = sigmaKf * snGrad(alpha1) (element-wise internal + boundary).
            fvcc::SurfaceField<NeoN::scalar> fSigma(
                exec, "fSigma", mesh,
                fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh)
            );
            fSigma.internalVector() = sigmaKf.internalVector();
            NeoN::mul(fSigma.internalVector(), snAlpha.internalVector());
            fSigma.boundaryData().value() = sigmaKf.boundaryData().value();
            NeoN::mul(fSigma.boundaryData().value(), snAlpha.boundaryData().value());
            return fSigma;
        },
        "runtime"_a,
        "alpha1"_a,
        "sigma"_a,
        "Surface-tension face force sigma*K*snGrad(alpha1) (interfaceProperties)"
    );
}
} // namespace NeoFOAM::bindings
