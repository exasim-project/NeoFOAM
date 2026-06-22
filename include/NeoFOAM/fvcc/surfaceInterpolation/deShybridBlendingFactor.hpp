// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

/**
 * @brief Coefficients of OpenFOAM's DEShybrid convection scheme (Travin et al. 2000;
 * Spalart et al. 2012). Defaults match the OpenFOAM reader (CH1=3, CH2=1, CH3=2, Cs=0.18 are
 * fixed scheme constants; the remainder are read from the fvSchemes spec).
 *
 * The OpenFOAM spec order is:
 *   DEShybrid <scheme1> <scheme2> <delta> CDES U0 L0 sigmaMin sigmaMax OmegaLim [nutLim]
 */
struct DEShybridCoefficients
{
    NeoN::scalar CDES = 0.65;    ///< DES coefficient
    NeoN::scalar U0 = 1.0;       ///< reference velocity scale [m/s] (> 0)
    NeoN::scalar L0 = 1.0;       ///< reference length scale [m] (> 0)
    NeoN::scalar sigmaMin = 0.0; ///< lower bound for sigma (0..1)
    NeoN::scalar sigmaMax = 1.0; ///< upper bound for sigma (0..1)
    NeoN::scalar OmegaLim = 1.0e-3; ///< limiter of the B function
    NeoN::scalar nutLim = 1.0;      ///< GAM-extension nut limiter (> 1 activates the extension)

    // Fixed scheme constants (Spalart et al. 2012).
    NeoN::scalar CH1 = 3.0;
    NeoN::scalar CH2 = 1.0;
    NeoN::scalar CH3 = 2.0;
    NeoN::scalar Cs = 0.18;
};

/**
 * @brief Computes the per-cell DEShybrid blending factor sigma from turbulence quantities.
 *
 * Implements the Travin et al. formula exactly (see OpenFOAM DEShybrid::calcBlendingFactor):
 *   S      = sqrt(2)*mag(symm(gradU)),  Omega = sqrt(2)*mag(skew(gradU)),  tau0 = L0/U0
 *   B      = CH3*Omega*max(S,Omega) / max(0.5*(S^2+Omega^2), (OmegaLim/tau0)^2)
 *   g      = tanh(B^4)
 *   K      = max(sqrt(0.5*(S^2+Omega^2)), 0.1/tau0)
 *   lTurb  = sqrt(max((max(nut, min((Cs*delta)^2*S, nutLim*nut)) + nu)/(0.09^1.5*K), 0))
 *   A      = CH2*max(0, CDES*delta/max(lTurb*g, SMALL*L0) - 0.5)
 *   sigma  = max(sigmaMax*tanh(A^CH1), sigmaMin)
 *
 * @param gradU      velocity gradient tensor (cell-centred)
 * @param nut        turbulent viscosity (cell-centred)
 * @param nu         laminar viscosity (cell-centred)
 * @param delta      LES filter width (cell-centred)
 * @param coeffs     DEShybrid coefficients
 * @param sigmaCell  output per-cell blending factor (internal field filled; sized to the mesh)
 */
void computeDEShybridSigmaCell(
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU,
    const nnfvcc::VolumeField<NeoN::scalar>& nut,
    const nnfvcc::VolumeField<NeoN::scalar>& nu,
    const nnfvcc::VolumeField<NeoN::scalar>& delta,
    const DEShybridCoefficients& coeffs,
    nnfvcc::VolumeField<NeoN::scalar>& sigmaCell
);

/**
 * @brief Convenience wrapper: computes the per-cell sigma and interpolates it to faces.
 *
 * @param gradU, nut, nu, delta, coeffs   as for computeDEShybridSigmaCell
 * @param surfInterp  surface interpolation used to map cell sigma to faces (e.g. linear)
 * @param sigmaFace   output per-face blending factor (the field NeoN's DEShybrid consumes)
 */
void computeDEShybridBlendingFactor(
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU,
    const nnfvcc::VolumeField<NeoN::scalar>& nut,
    const nnfvcc::VolumeField<NeoN::scalar>& nu,
    const nnfvcc::VolumeField<NeoN::scalar>& delta,
    const DEShybridCoefficients& coeffs,
    const nnfvcc::SurfaceInterpolation<NeoN::scalar>& surfInterp,
    nnfvcc::SurfaceField<NeoN::scalar>& sigmaFace
);

/**
 * @brief Reads the DEShybrid coefficients from a fvSchemes div-scheme token list.
 *
 * Accepts the OpenFOAM spec (after passthrough by mapFvSchemes), e.g.
 *   Gauss DEShybrid linear linearUpwind grad(U) delta 0.65 30 2 0 1 1.0e-03 1.0
 * The trailing numeric tokens are the coefficients
 *   CDES U0 L0 sigmaMin sigmaMax OmegaLim [nutLim]
 * read robustly regardless of whether each was tokenised as a label or a scalar (OpenFOAM stores
 * integer-looking values such as "30" or "0" as labels). The fixed scheme constants
 * (CH1, CH2, CH3, Cs) keep their defaults. The LES-delta field name is ignored here: the LES filter
 * width comes from the turbulence model's own delta.
 *
 * @throws if fewer than 6 trailing numeric coefficients are present.
 */
DEShybridCoefficients readDEShybridCoefficients(NeoN::TokenList divSchemeTokens);

} // namespace NeoFOAM
