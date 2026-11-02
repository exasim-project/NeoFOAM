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
 * The face-normal gradient scheme is resolved from the snGradSchemes subdict of fvSchemes,
 * as fvc::snGrad(p) does. Applied to internal and non-processor boundary faces.
 */
void addConsistentFluxCorrection(
    nnfvcc::SurfaceField<scalar>& phiHbyA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& rAtU,
    const nnfvcc::VolumeField<scalar>& p,
    const NeoN::Dictionary& fvSchemes
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

/* @brief Which overall sign convention the caller's pEqn was assembled with.
 *
 * LaplacianMinusDiv: pEqn built as +laplacian(...) - div(...) == 0, mirroring OpenFOAM's
 *   pimpleFoam/simpleFoam-style `fvm::laplacian(rAtU, p) == fvc::div(phiHbyA)`. This is
 *   what NeoFOAM's incompressible solvers (and the "compute flux" unit test) build, and
 *   is the default.
 * DivMinusLaplacian: pEqn built as +div(...) - laplacian(...) == 0, mirroring OpenFOAM's
 *   rhoPimpleFoam-style `pDDtEqn - fvm::laplacian(rhorAUf, p)`. The compressible solver
 *   uses this form because its ddt(psi,p) term then also contributes a positive diagonal,
 *   keeping the matrix positive-definite -- required for Cholesky-family (DIC/IC)
 *   preconditioning of p, matching literal rhoPimpleFoam's own solver choice.
 *
 * The two forms assemble pEqn with exactly opposite-signed matrix coefficients for the
 * same physical equation, so updateFaceVelocity needs to know which one it was given in
 * order to reconstruct the correctly-signed face flux.
 */
enum class PEqnConvention
{
    LaplacianMinusDiv,
    DivMinusLaplacian
};

/* @brief Updates the face flux: phi = phiHbyA - pEqn.flux() (sign per `convention`).
 *
 * pEqn.flux() includes both the orthogonal matrix-coefficient flux and the non-orthogonal
 * face flux correction stashed in the linear system. Without the correction div(phi) != 0
 * on non-orthogonal meshes. Requires an assembled pressure system with
 * keepFaceFluxCorrection(true).
 */
void updateFaceVelocity(
    const nnfvcc::SurfaceField<scalar>& predictedPhi,
    const PDE<scalar>& expr,
    nnfvcc::SurfaceField<scalar>& phi,
    PEqnConvention convention = PEqnConvention::LaplacianMinusDiv
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

/* @brief Velocity corrector using the gradSchemes entry configured for @p p.
 *
 * Resolves "grad(<p.name>)" through RunTime::gradScheme, which builds and caches the
 * operator, so solvers need not construct one themselves.
 */
void updateVelocity(
    const nnfvcc::VolumeField<Vec3>& hByA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& p,
    nnfvcc::VolumeField<Vec3>& U,
    RunTime& runTime
);

/* @brief Face flux from a volume vector field: phi = U·Sf.
 */
nnfvcc::SurfaceField<scalar> flux(const nnfvcc::VolumeField<Vec3>& volField);

}
