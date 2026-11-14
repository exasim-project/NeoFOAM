// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/runTime.hpp"

namespace NeoFOAM
{

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

/* @brief The buoyancy head gh = (g & C) - ghRef of gh.H, on cells and boundary faces.
 *
 * @detail g is read from constant/g and hRef from constant/hRef (0 when absent, as readhRef.H);
 * ghRef = g & (cmptMag(g)/|g|)*hRef. Evaluated by OpenFOAM and copied, so it is OpenFOAM's
 * field bit for bit. Setup only: a static mesh never needs it again.
 */
nnfvcc::VolumeField<NeoN::scalar> createGh(RunTime& rt);

/* @brief The face buoyancy head ghf = (g & Cf) - ghRef of gh.H (see createGh). */
nnfvcc::SurfaceField<NeoN::scalar> createGhf(RunTime& rt);

/* @brief The gravity face force -ghf*snGrad(rho) of interFoam's UEqn.H / pEqn.H, into @p force.
 *
 * @detail The term `mixture.surfaceTensionForce() - ghf*fvc::snGrad(rho)` without surface
 * tension (sigma == 0 makes the first term a signed zero, which leaves the difference exact).
 * snGrad follows fvc::snGrad with the snGradSchemes entry for rho: "uncorrected", or "corrected"
 * with the non-orthogonal correction from grad(rho) under "Gauss linear" (other schemes throw).
 * Internal faces use OpenFOAM's nonOrthDeltaCoeffs and weights (foamGeometry), so Serial matches
 * OpenFOAM bit for bit; boundary faces take the patch snGrad (see buoyantMomentumFlux), which
 * for rho's calculated patches is deltaCoeffs*(rho_b - rho_P). Coupled patches throw.
 */
void gravityFaceForce(
    RunTime& rt,
    const nnfvcc::SurfaceField<NeoN::scalar>& ghf,
    const nnfvcc::VolumeField<NeoN::scalar>& rho,
    nnfvcc::SurfaceField<NeoN::scalar>& force
);

/* @brief pEqn.H's buoyancy flux phig = force*rAUf*magSf (internal and boundary faces).
 *
 * @p force is gravityFaceForce's result.
 */
void buoyancyFlux(
    const nnfvcc::SurfaceField<NeoN::scalar>& force,
    const nnfvcc::SurfaceField<NeoN::scalar>& rAUf,
    nnfvcc::SurfaceField<NeoN::scalar>& phig
);

/* @brief UEqn.H's momentum-source face flux (force - snGrad(p_rgh))*magSf, into @p flux.
 *
 * @detail fvc::reconstruct of it (see reconstruct) is the right-hand side of the momentum
 * predictor. snGrad(p_rgh) as in gravityFaceForce, except that the boundary takes the patch
 * snGrad by type: refGrad on fixedGradient and fixedFluxPressure patches (OpenFOAM's
 * fixedGradient::snGrad returns the gradient), deltaCoeffs*(p_b - p_P) elsewhere.
 */
void buoyantMomentumFlux(
    RunTime& rt,
    const nnfvcc::SurfaceField<NeoN::scalar>& force,
    const nnfvcc::VolumeField<NeoN::scalar>& p_rgh,
    nnfvcc::SurfaceField<NeoN::scalar>& flux
);

/* @brief fvc::reconstruct(ssf) into @p out, as OpenFOAM evaluates it.
 *
 * @detail inv(surfaceSum(SfHat*Sf)) & surfaceSum(SfHat*ssf) with SfHat = Sf/magSf, the full
 * (non-symmetric) tensor and Tensor::safeInv's handling of 2-D meshes. Every cell gathers its
 * faces in fvc::surfaceSum's order (cellFaces), so the result is OpenFOAM's on Serial and CPU
 * bit for bit. Boundary values are the owner-cell values (extrapolatedCalculated). Coupled
 * patches throw. NeoN's reconstruct regularises 2-D meshes differently.
 */
void reconstruct(
    const nnfvcc::SurfaceField<NeoN::scalar>& ssf,
    nnfvcc::VolumeField<NeoN::Vec3>& out
);

/* @brief interFoam's velocity correction U = HbyA + rAU*reconstruct((phig - pFlux)/rAUf).
 *
 * @detail pEqn.H's `U = HbyA + rAU()*fvc::reconstruct((phig - p_rghEqn.flux())/rAUf)`, with
 * @p pFlux the pressure equation's face flux p_rghEqn.flux(). Unlike updateVelocity
 * (U = HbyA - rAU*grad(p)) it carries the buoyancy flux, which the pressure flux balances at
 * the walls; the plain form leaves spurious velocities there. The reconstruct argument is formed
 * face by face inside the gather (see reconstruct), so no surface field is allocated. Only the
 * internal values of @p U are written: call U.correctBoundaryConditions() afterwards, as pEqn.H
 * does.
 */
void updateVelocityBuoyant(
    const nnfvcc::VolumeField<NeoN::Vec3>& hByA,
    const nnfvcc::VolumeField<NeoN::scalar>& rAU,
    const nnfvcc::SurfaceField<NeoN::scalar>& phig,
    const nnfvcc::SurfaceField<NeoN::scalar>& pFlux,
    const nnfvcc::SurfaceField<NeoN::scalar>& rAUf,
    nnfvcc::VolumeField<NeoN::Vec3>& U
);

/* @brief The static pressure p == p_rgh + rho*gh of pEqn.H (internal and boundary values).
 *
 * @detail Without the pressure-reference levelling pEqn.H adds when p_rgh needs a reference.
 */
void updateStaticPressure(
    const nnfvcc::VolumeField<NeoN::scalar>& p_rgh,
    const nnfvcc::VolumeField<NeoN::scalar>& rho,
    const nnfvcc::VolumeField<NeoN::scalar>& gh,
    nnfvcc::VolumeField<NeoN::scalar>& p
);

} // namespace NeoFOAM
