// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "NeoN/NeoN.hpp"

namespace NeoFOAM
{

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

/* @brief Bounded explicit MULES solve, Foam::MULES::explicitSolve (MULESTemplates.C).
 *
 * @detail rho == 1, Sp == Su == 0, static mesh, Euler time step, extremaCoeff ==
 * boundaryExtremaCoeff == smoothLimiter == 0. As in OpenFOAM, @p alpha (psi, the current
 * corrector's field) gives the upwind flux phiBD and the local extrema, while @p alphaOld
 * (psi.oldTime(), alpha at the start of the time step) enters the bound right-hand sides and
 * the update alpha = (alphaOld*rDeltaT - surfaceIntegrate(alphaPhi))/rDeltaT. Both are the same
 * field for the first corrector (nAlphaCorr == 1); @p alphaOld may alias @p alpha.
 * @p phi is the volumetric flux: it defines the upwind (donor) flux and is the BoundaryContext
 * "phi" for alpha.correctBoundaryConditions, which runs on entry and exit (as in OpenFOAM).
 * Limits @p alphaPhi in place (high-order flux in, limited flux out) and advances @p alpha
 * conservatively.
 *
 * Boundary treatment (MULES::limiter): patches that fix the value (NeoN fixesValue, plus
 * inletOutlet, which is a mixed patch and fixesValue() in OpenFOAM) widen the local extrema of
 * their face cells by the patch value. The boundary correction flux is zero (phiBD_b ==
 * alphaPhi_b), so the boundary limiter is irrelevant. Coupled (processor) patches throw; wedge
 * patches do not exist in NeoN.
 */
void mulesExplicitSolve(
    nnfvcc::VolumeField<NeoN::scalar>& alpha,
    const nnfvcc::VolumeField<NeoN::scalar>& alphaOld,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi,
    nnfvcc::SurfaceField<NeoN::scalar>& alphaPhi,
    NeoN::scalar deltaT,
    NeoN::scalar psiMax,
    NeoN::scalar psiMin,
    int nLimiterIter
);

/* @brief MULES::correct (CMULESTemplates.C) for the VoF path, psiMax == 1, psiMin == 0.
 *
 * @detail Limits the correction flux @p phiCorr (internal and boundary faces) so that applying
 * it keeps alpha bounded, then applies it: alpha = (alpha*rDeltaT - surfaceIntegrate(phiCorr))
 * / rDeltaT. @p phi plays the role of OpenFOAM's limiterCorr phi argument (interFoam passes
 * alphaPhiUn): a boundary face is limited only where phi_b + phiCorr_b > SMALL^2 (outlets).
 * Fixed-value / inletOutlet patches widen the local extrema as in mulesExplicitSolve; coupled
 * patches throw. @p relaxOld blends the result with the pre-correction field (interFoam
 * aCorr > 0 branch, 0.5); 0 disables it.
 *
 * Unlike OpenFOAM, the boundary values of @p alpha are NOT corrected here: an inletOutlet patch
 * needs the volumetric flux, which @p phi is not. Call correctBoundaryConditions afterwards.
 */
void mulesCorrect(
    nnfvcc::VolumeField<NeoN::scalar>& alpha,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi,
    nnfvcc::SurfaceField<NeoN::scalar>& phiCorr,
    double deltaT,
    int nLimiterIter,
    double relaxOld
);

/* @brief alpha.correctBoundaryConditions with the volumetric flux @p phi in the
 * BoundaryContext under "phi", so inletOutlet patches switch per face on the flux sign. */
void correctBoundaryConditionsWithFlux(
    nnfvcc::VolumeField<NeoN::scalar>& alpha,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi
);

} // namespace NeoFOAM
