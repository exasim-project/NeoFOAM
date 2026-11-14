// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/runTime.hpp"

namespace NeoFOAM
{

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

/* @brief The constant/transportProperties of an incompressible two-phase (VoF) case. */
struct TwoPhaseTransportProperties
{
    std::string phase1; ///< first entry of `phases`; alpha1 is alpha.<phase1>
    std::string phase2;
    NeoN::scalar rho1;
    NeoN::scalar nu1;
    NeoN::scalar rho2;
    NeoN::scalar nu2;
    NeoN::scalar sigma;
};

/* @brief Read constant/transportProperties as incompressibleTwoPhaseMixture does.
 *
 * @detail Both phases must use the Newtonian viscosity model and sigma must be a constant
 * (the constant surfaceTensionModel); anything else throws, since the mixture operators below
 * only implement those.
 */
TwoPhaseTransportProperties readTwoPhaseTransportProperties(RunTime& rt);

/* @brief rho = alpha1*rho1 + alpha2*rho2 with alpha2 = 1 - alpha1, as alphaEqnSubCycle.H's
 * `rho == alpha1*rho1 + alpha2*rho2` (internal and boundary values, alpha1 not clamped). */
void updateMixtureDensity(
    const nnfvcc::VolumeField<NeoN::scalar>& alpha1,
    NeoN::scalar rho1,
    NeoN::scalar rho2,
    nnfvcc::VolumeField<NeoN::scalar>& rho
);

/* @brief Mixture kinematic viscosity of incompressibleTwoPhaseMixture::calcNu (Newtonian phases).
 *
 * @detail nu = mu/(a*rho1 + (1 - a)*rho2) with mu = a*rho1*nu1 + (1 - a)*rho2*nu2 and
 * a = clamp(alpha1, 0, 1), in OpenFOAM's evaluation order, on the internal and boundary values.
 * interFoam's laminar stress uses rho*nu (divDevRhoReff), not the mixture's mu.
 */
void updateMixtureViscosity(
    const nnfvcc::VolumeField<NeoN::scalar>& alpha1,
    const TwoPhaseTransportProperties& properties,
    nnfvcc::VolumeField<NeoN::scalar>& nu
);

/* @brief out = x*y on the internal and boundary values; @p out may be @p x or @p y.
 *
 * @detail The field products of interFoam's momentum and pressure equations (rho*nuEff,
 * rho*rAU, interpolate(rho)*phi) without a temporary field per product.
 */
void multiply(
    const nnfvcc::VolumeField<NeoN::scalar>& x,
    const nnfvcc::VolumeField<NeoN::scalar>& y,
    nnfvcc::VolumeField<NeoN::scalar>& out
);

/* @brief multiply of surface fields. */
void multiply(
    const nnfvcc::SurfaceField<NeoN::scalar>& x,
    const nnfvcc::SurfaceField<NeoN::scalar>& y,
    nnfvcc::SurfaceField<NeoN::scalar>& out
);

/* @brief fvc::interpolate(vf) under "linear" with OpenFOAM's weights, into @p out.
 *
 * @detail Internal faces w*(P - N) + N with the weights cached by foamGeometry, so Serial and CPU
 * match OpenFOAM bit for bit (NeoN's own linear interpolation computes the weights differently);
 * boundary faces take the patch values. Coupled (processor) patches throw.
 */
void linearInterpolate(
    RunTime& rt,
    const nnfvcc::VolumeField<NeoN::scalar>& vf,
    nnfvcc::SurfaceField<NeoN::scalar>& out
);

} // namespace NeoFOAM
