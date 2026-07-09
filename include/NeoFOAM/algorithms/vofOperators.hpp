// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/runTime.hpp"

namespace NeoFOAM
{

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

/* @brief Interface unit-normal flux nHatf = nHatfv & Sf.
 *
 * @detail With nHatfv = gradAlphaf / (|gradAlphaf| + deltaN),
 * gradAlphaf = interpolate(grad(alpha1)) (Gauss-Green grad, linear face interp),
 * deltaN = 1e-8 / cbrt(mean(cell volume)). Faithful to
 * interfaceProperties::calculateK (interfaceProperties.C:108-166) minus the
 * contact-angle correction and curvature smoothing. Scatters internal + boundary
 * faces via Kokkos parallelFor over the mesh Sf views. Single source of truth for
 * the interface normal flux, reused by alphaPhaseFlux and surfaceTensionForce.
 */
nnfvcc::SurfaceField<NeoN::scalar> computeNHatf(const nnfvcc::VolumeField<NeoN::scalar>& alpha1);

/* @brief fvc::flux(phi, alpha, "Gauss vanLeer"): the vanLeer-limited convective
 * face flux phi_f * alpha_f.
 *
 * @detail alpha_f uses the limited interpolation weight
 *   w_f     = limiter*wCD + (1 - limiter)*pos0(phi_f)
 *   wCD     = SfdNei/(SfdOwn + SfdNei)                        (linear/CD weight)
 *   limiter = (r + |r|)/(1 + |r|)                             (vanLeerLimiter)
 *   r       = NVDTVD::r(phi_f, alpha_own, alpha_nei, gradc_own, gradc_nei, d)
 *   gradc   = fvc::grad(alpha) (Gauss linear == GaussGreenGrad), d = C_nei - C_own
 * Faithful transcription of Foam::LimitedScheme::calcLimiter + NVDTVD::r +
 * vanLeerLimiter. Boundary flux is phi_b * alpha_b (non-coupled patch value).
 */
nnfvcc::SurfaceField<NeoN::scalar> vanLeerAlphaFlux(
    RunTime& rt,
    const nnfvcc::VolumeField<NeoN::scalar>& alpha,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi
);

/* @brief interFoam alphaEqn.H high-order phase flux alphaPhiUn.
 *
 * @detail alphaPhiUn = fvc::flux(phi, alpha1, "div(phi,alpha)")           [vanLeer]
 *   + fvc::flux(-fvc::flux(-phir, alpha2, ...), alpha1, ...)         [compression]
 * with phir = phic*nHatf, phic = cAlpha*|phi/magSf| (non-coupled boundary phic == 0),
 * alphaScheme = Gauss vanLeer, alpharScheme = Gauss linear. Faithful to VoF/alphaEqn.H
 * (icAlpha == scAlpha == 0, the damBreak / interFoam default).
 */
nnfvcc::SurfaceField<NeoN::scalar> alphaPhaseFlux(
    RunTime& rt,
    const nnfvcc::VolumeField<NeoN::scalar>& alpha1,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi,
    double cAlpha
);

/* @brief FCT lambda limiter sweep shared by mulesExplicitSolve and mulesCorrect.
 *
 * @detail Foam::MULES::limiter inner sweep: given the per-cell allowable bound RHS
 * (@p psiMaxn, @p psiMinn) and the antidiffusive-flux extrema sums (@p sumPhip,
 * @p mSumPhim), iterate @p nLimiterIter times to compute the per-internal-face
 * limiter lambda in [0, 1] for the antidiffusive flux @p phiCorr. Returns the
 * lambda vector (nInternalFaces, initialized to 1). Boundary lambda stays 1.
 */
NeoN::Vector<NeoN::scalar> computeFCTLambda(
    const NeoN::Executor& exec,
    const NeoN::UnstructuredMesh& mesh,
    NeoN::View<const NeoN::scalar> phiCorr,
    NeoN::View<const NeoN::scalar> psiMaxn,
    NeoN::View<const NeoN::scalar> psiMinn,
    NeoN::View<const NeoN::scalar> sumPhip,
    NeoN::View<const NeoN::scalar> mSumPhim,
    int nLimiterIter
);

/* @brief Bounded explicit MULES (FCT) solve — Foam::MULES::explicitSolve.
 *
 * @detail For the simplified VoF path (rho == 1, Sp == Su == 0, static mesh,
 * extremaCoeff == smoothLimiter == 0, no coupled/fixed-value boundary widening).
 * Limits @p alphaPhi in place (high-order flux in, FCT-limited out) and advances
 * @p alpha conservatively (no clamp — boundedness comes from the limiter). @p alpha
 * is the current == oldTime field (psi0) at entry.
 */
void mulesExplicitSolve(
    nnfvcc::VolumeField<NeoN::scalar>& alpha,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi,
    nnfvcc::SurfaceField<NeoN::scalar>& alphaPhi,
    NeoN::scalar deltaT,
    NeoN::scalar psiMax,
    NeoN::scalar psiMin,
    int nLimiterIter
);

/* @brief MULES::correct (CMULESTemplates.C limiterCorr + correct) for the VoF path.
 *
 * @detail rho==1, Sp==Su==0, extremaCoeff==smoothLimiter==0, psiMax==1, psiMin==0:
 * FCT-limit the antidiffusive correction flux @p phiCorr so applying it keeps alpha
 * bounded, then apply it — alpha := alpha - deltaT*surfaceIntegrate(phiCorr). @p alpha
 * is the predictor-advanced field on entry; both @p alpha and @p phiCorr are modified
 * in place. @p relaxOld under-relaxes the corrected field against its pre-correction
 * state (interFoam aCorr>0 branch); relaxOld = 0 reproduces the first-corrector
 * behaviour (no under-relaxation).
 */
void mulesCorrect(
    nnfvcc::VolumeField<NeoN::scalar>& alpha,
    nnfvcc::SurfaceField<NeoN::scalar>& phiCorr,
    double deltaT,
    int nLimiterIter,
    double relaxOld
);

/* @brief Surface-tension face force fSigma = interpolate(sigma*K)*snGrad(alpha1).
 *
 * @detail Curvature K = -div(nHatf) = -surfaceIntegrate(nHatf). Mirrors
 * interfaceProperties::surfaceTensionForce (interfaceProperties.C:237-241).
 *
 * @warning Orthogonal meshes only — uses the uncorrected snGrad scheme
 * (uncorrected == corrected on an orthogonal mesh). On a non-orthogonal mesh the
 * missing explicit non-orthogonal correction makes this silently wrong.
 */
nnfvcc::SurfaceField<NeoN::scalar>
surfaceTensionForce(const nnfvcc::VolumeField<NeoN::scalar>& alpha1, double sigma);

} // namespace NeoFOAM
