// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 FoamAdapter authors

#pragma once

#include "FoamAdapter/datastructures/expression.hpp"

#include "NeoN/NeoN.hpp"

namespace nnfvcc = NeoN::finiteVolume::cellCentred;
using scalar = NeoN::scalar;
using Vec3 = NeoN::Vec3;

namespace FoamAdapter
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
nnfvcc::VolumeField<scalar> computeRAU(const PDESolver<Vec3>& expr);

/* @brief given access to a PDESolver this function computes rAU and HbyA
 * from the assembled system
 *
 * where rAU  - inverse of the system matrix diagonal
 *       HbyA - offdiagonal entries divided by diagonal
 *
 * @return a tuple containing rAU and HbyA
 */
std::tuple<nnfvcc::VolumeField<scalar>, nnfvcc::VolumeField<Vec3>>
computeRAUandHByA(const PDESolver<Vec3>& expr);

/* @brief computes phi = phiHbyA - pEqn.flux();
 * where pEqn.flux
 * @note assumes an assembled system matrix
 */
void updateFaceVelocity(
    const nnfvcc::SurfaceField<scalar>& predictedPhi,
    const PDESolver<scalar>& expr,
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
