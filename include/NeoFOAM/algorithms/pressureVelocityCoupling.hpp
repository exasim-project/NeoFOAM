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

/* @brief ensure that the HbyA does not violate boundary velocity constraint
 *
 * @detail
 * See https://openfoamwiki.net/index.php/SimpleFoam for details
 *
 * Cite:
 * The velocity at the boundary face should satisfy following equation:
 * \bold{u}|_{bf} = \frac{\bold {H[u] }}{a_P }|_{bf} - \frac{\nabla p{}}{a_P }|_{bf}
 * The subscript bf denoted that the quantity is evaluated at the boundary face. The function
 * constrainHbyA ensures that the field <math> \frac{\bold {H[u] }}{a_P }|_{bf} </math> does not
 * violate the above equation. The boundary condition fixedFluxExtrapolatedPressure sets the
 * pressure gradient in order that the above equation is satisfied. If we cannot modify the
 * velocity, the function sets the field <math> \frac{\bold {H[u] }}{a_P }|_{bf}  = \bold{u}|_{bf}
 * </math> in order that the field  <math> \frac{\bold {H[u] }}{a_P }|_{bf} </math> does not
 * contradict the zero gradient boundary condition which should be applied for the pressure if the
 * velocity is fixed.
 */
void constrainHbyA(
    const nnfvcc::VolumeField<Vec3>& U,
    const nnfvcc::VolumeField<scalar>& p,
    nnfvcc::VolumeField<Vec3>& HbyA
);

/* @brief given a ... this function computes rAU
 *
 * where rAU  - inverse of the system matrix diagonal
 *
 * @return a tuple containing rAU and HbyA
 */
nnfvcc::VolumeField<scalar> computeRAU(const PDE<Vec3>& expr);

/* @brief given access to a PDE this function computes rAU and HbyA
 * from the assembled system
 *
 * where rAU  - inverse of the system matrix diagonal
 *       HbyA - offdiagonal entries divided by diagonal
 *
 * @return a tuple containing rAU and HbyA
 */
std::tuple<nnfvcc::VolumeField<scalar>, nnfvcc::VolumeField<Vec3>>
computeRAUandHByA(const PDE<Vec3>& expr);

/* @brief consistent (SIMPLEC) reciprocal diagonal rAtU
 *
 * @details Implements OpenFOAM's `rAtU = 1/(1/rAU - UEqn.H1())` (simpleFoam/pEqn.H,
 * enabled by `SIMPLE/consistent yes`). H1 is the OpenFOAM fvMatrix::H1() analogue:
 * the negated sum of the off-diagonal coefficients of a row divided by the cell
 * volume (lduMatrix::H1 subtracts every off-diagonal; fvMatrix::H1 then divides by V
 * and folds in the coupled-patch coupling). With A = 1/rAU = diag/V the denominator
 * becomes (diag + sum_offDiag)/V, so the consistent diagonal absorbs the neighbour
 * coupling that plain SIMPLE drops — letting the pressure equation stay consistent at
 * an under-relaxation factor of 1.
 *
 * @note assumes an assembled system matrix; the off-diagonal coupling of processor
 * boundary faces is folded in the same way computeRAUandHByA handles it.
 */
nnfvcc::VolumeField<scalar>
computeRAtU(const PDESolver<Vec3>& expr, const nnfvcc::VolumeField<scalar>& rAU);

/* @brief SIMPLEC flux correction: phiHbyA += interpolate(rAtU - rAU)*snGrad(p)*magSf
 *
 * @details Mirrors the `phiHbyA += fvc::interpolate(rAtU() - rAU)*fvc::snGrad(p)*mesh.magSf()`
 * term in simpleFoam/pEqn.H. snGrad(p) uses the corrected face-normal gradient so the
 * non-orthogonal part is retained. Applied to internal and boundary faces.
 */
void addConsistentFluxCorrection(
    nnfvcc::SurfaceField<scalar>& phiHbyA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& rAtU,
    const nnfvcc::VolumeField<scalar>& p
);

/* @brief SIMPLEC HbyA correction: HbyA -= (rAU - rAtU)*grad(p)
 *
 * @details Mirrors `HbyA -= (rAU - rAtU())*fvc::grad(p)` in simpleFoam/pEqn.H. Combined
 * with the velocity corrector U = HbyA - rAtU*grad(p) this reproduces the plain-SIMPLE
 * reconstruction U = HbyA0 - rAU*grad(p) at convergence (where p has stopped changing).
 */
void subtractConsistentHbyA(
    nnfvcc::VolumeField<Vec3>& hByA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& rAtU,
    const nnfvcc::VolumeField<scalar>& p
);

/* @brief computes phi = phiHbyA - pEqn.flux();
 * where pEqn.flux() = (orthogonal matrix-coefficient flux) + faceFluxCorrection
 *
 * @detail The pressure Laplacian defers its non-orthogonal snGrad correction to the matrix
 * RHS (deferred correction) and stashes the per-face correction flux in the linear system
 * (LinearSystem::faceFluxCorrection(), the OpenFOAM fvMatrix::faceFluxCorrectionPtr_ analogue).
 * This reconstruction adds it back; the orthogonal-only reconstruction would otherwise leave
 * div(phi) = div(correctionFlux) != 0, inflating the continuity error on non-orthogonal meshes
 * while orthogonal / uncorrected meshes stay correct.
 *
 * @note assumes an assembled system matrix
 */
void updateFaceVelocity(
    const nnfvcc::SurfaceField<scalar>& predictedPhi,
    const PDE<scalar>& expr,
    nnfvcc::SurfaceField<scalar>& phi
);

/* @brief velocity based on HbyA, rAU and current pressure value
 * U = HbyA - rAU*fvc::grad(p);
 *
 * @details once Hby rAU and a current pressure value is available
 * an updated velocity can be computed according to
 * \bold {u_P} = \frac{\bold {H[u^*] }}{a_P^* } - \frac{1}{a_P^* }\nabla p_P
 * See https://openfoamwiki.net/index.php/SimpleFoam for details
 */
void updateVelocity(
    const nnfvcc::VolumeField<Vec3>& hByA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& p,
    nnfvcc::VolumeField<Vec3>& U
);


/* @brief Reimplementation of OpenFOAMs fvMatrix.flux()
 * @return flux surface field
 */
nnfvcc::SurfaceField<scalar> flux(const nnfvcc::VolumeField<Vec3>& volField);

}
