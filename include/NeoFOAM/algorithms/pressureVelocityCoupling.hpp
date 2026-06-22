// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoFOAM/datastructures/pde.hpp"

#include "NeoN/NeoN.hpp"

namespace nnfvcc = NeoN::finiteVolume::cellCentred;
using scalar = NeoN::scalar;
using Vec3 = NeoN::Vec3;

namespace NeoFOAM
{

/* @brief Ensures HbyA satisfies velocity boundary constraints.
 *
 * On fixed-value velocity patches sets HbyA = U so downstream pressure
 * reconstruction does not contradict the imposed velocity. On fixed-flux
 * patches the pressure gradient is adjusted instead and HbyA is unchanged.
 */
void constrainHbyA(
    const nnfvcc::VolumeField<Vec3>& U,
    const nnfvcc::VolumeField<scalar>& p,
    nnfvcc::VolumeField<Vec3>& HbyA
);

/* @brief Inverse of the assembled momentum matrix diagonal: rAU = 1/diag.
 */
nnfvcc::VolumeField<scalar> computeRAU(const PDE<Vec3>& expr);

/* @brief Inverse diagonal (rAU) and off-diagonal source (HbyA) of the assembled momentum matrix.
 *
 * rAU  = 1/diag
 * HbyA = offDiag/diag
 */
std::tuple<nnfvcc::VolumeField<scalar>, nnfvcc::VolumeField<Vec3>>
computeRAUandHByA(const PDE<Vec3>& expr);

/* @brief SIMPLEC consistent reciprocal diagonal: rAtU = 1/(1/rAU + sumOffDiag/V).
 *
 * Absorbs off-diagonal neighbour coupling into the diagonal, letting the pressure
 * equation remain consistent with an under-relaxation factor of 1.
 * Requires an assembled momentum system; processor-boundary coupling is folded in
 * via the off-diagonal matrix.
 */
nnfvcc::VolumeField<scalar>
computeRAtU(const PDESolver<Vec3>& expr, const nnfvcc::VolumeField<scalar>& rAU);

/* @brief SIMPLEC flux correction: phiHbyA += interpolate(rAtU - rAU)*snGrad(p)*magSf.
 *
 * snGrad(p) uses the corrected face-normal gradient to retain non-orthogonal contributions.
 * Applied to internal and non-processor boundary faces.
 */
void addConsistentFluxCorrection(
    nnfvcc::SurfaceField<scalar>& phiHbyA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& rAtU,
    const nnfvcc::VolumeField<scalar>& p
);

/* @brief SIMPLEC HbyA correction: hByA -= (rAU - rAtU)*grad(p).
 *
 * At convergence, combined with U = HbyA - rAtU*grad(p), this recovers
 * U = HbyA0 - rAU*grad(p), matching the plain-SIMPLE velocity corrector.
 */
void subtractConsistentHbyA(
    nnfvcc::VolumeField<Vec3>& hByA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& rAtU,
    const nnfvcc::VolumeField<scalar>& p
);

/* @brief Updates the face flux: phi = phiHbyA - pEqn.flux().
 *
 * pEqn.flux() includes both the orthogonal matrix-coefficient flux and the non-orthogonal
 * face flux correction stashed in the linear system. Without the correction div(phi) != 0
 * on non-orthogonal meshes. Requires an assembled pressure system with
 * keepFaceFluxCorrection(true).
 */
void updateFaceVelocity(
    const nnfvcc::SurfaceField<scalar>& predictedPhi,
    const PDE<scalar>& expr,
    nnfvcc::SurfaceField<scalar>& phi
);

/* @brief Velocity corrector: U = HbyA - rAU*grad(p).
 */
void updateVelocity(
    const nnfvcc::VolumeField<Vec3>& hByA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& p,
    nnfvcc::VolumeField<Vec3>& U,
    const nnfvcc::GradOperatorFactory<NeoN::Vec3>& gradPScheme
);


/* @brief Face flux from a volume vector field: phi = U·Sf.
 */
nnfvcc::SurfaceField<scalar> flux(const nnfvcc::VolumeField<Vec3>& volField);

}
