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
 * @detail nHatfv = gradAlphaf / (|gradAlphaf| + deltaN), with
 * gradAlphaf = interpolate(grad(alpha1)) (Gauss-Green grad, linear face interpolation) and
 * deltaN = 1e-8 / cbrt(mean(cell volume)). Follows interfaceProperties::calculateK
 * without the contact-angle correction. The gradient is summed in OpenFOAM's face order with
 * OpenFOAM's linear weights, so Serial results match OpenFOAM to the last bit even where
 * gradAlpha is round-off, which the 1/deltaN normalisation amplifies. Boundary values use the
 * gaussGrad boundary correction.
 *
 * The weights (and vanLeerAlphaFlux's owner-to-neighbour deltas) are copied from the OpenFOAM
 * mesh in @p rt to the device once and cached in the NeoN mesh's stencil database.
 */
nnfvcc::SurfaceField<NeoN::scalar>
computeNHatf(RunTime& rt, const nnfvcc::VolumeField<NeoN::scalar>& alpha1);

/* @brief computeNHatf into an existing field (internal and boundary values overwritten). */
void computeNHatf(
    RunTime& rt,
    const nnfvcc::VolumeField<NeoN::scalar>& alpha1,
    nnfvcc::SurfaceField<NeoN::scalar>& nHatf
);

/* @brief fvc::flux(phi, alpha, "Gauss vanLeer"): vanLeer-limited face flux phi_f * alpha_f.
 *
 * @detail Transcribes Foam::LimitedScheme::calcLimiter + NVDTVD::r + vanLeerLimiter and
 * surfaceInterpolationScheme::interpolate (alpha_f = w*(alpha_P - alpha_N) + alpha_N).
 * Linear weights and the cell-to-cell delta are OpenFOAM's (cached per mesh, see computeNHatf).
 * Boundary flux is phi_b * alpha_b (non-coupled patch value).
 */
nnfvcc::SurfaceField<NeoN::scalar> vanLeerAlphaFlux(
    RunTime& rt,
    const nnfvcc::VolumeField<NeoN::scalar>& alpha,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi
);

/* @brief vanLeerAlphaFlux into an existing field (internal and boundary values overwritten). */
void vanLeerAlphaFlux(
    RunTime& rt,
    const nnfvcc::VolumeField<NeoN::scalar>& alpha,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi,
    nnfvcc::SurfaceField<NeoN::scalar>& flux
);

/* @brief Upwind face flux phi_f * alpha_upwind(f) as upwind<scalar>::flux computes it;
 * boundary phi_b * alpha_b. */
nnfvcc::SurfaceField<NeoN::scalar> upwindAlphaFlux(
    const nnfvcc::VolumeField<NeoN::scalar>& alpha,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi
);

/* @brief alphaEqn.H interface compression velocity phic = cAlpha*|phi/magSf|.
 *
 * @detail Zero on the boundary faces: interFoam zeroes phic on non-coupled patches, and NeoN
 * has no coupled ones.
 */
nnfvcc::SurfaceField<NeoN::scalar>
interfaceCompressionVelocity(const nnfvcc::SurfaceField<NeoN::scalar>& phi, NeoN::scalar cAlpha);

/* @brief interfaceCompressionVelocity into an existing field. */
void interfaceCompressionVelocity(
    const nnfvcc::SurfaceField<NeoN::scalar>& phi,
    NeoN::scalar cAlpha,
    nnfvcc::SurfaceField<NeoN::scalar>& phic
);

/* @brief alphaEqn.H compression flux fvc::flux(-fvc::flux(-phir, alpha2, linear), alpha1, linear).
 *
 * @detail alpha2 = 1 - alpha1. Internal faces: (phir*alpha2_f)*alpha1_f with OpenFOAM's linear
 * weights (cached per mesh, see computeNHatf), so Serial matches OpenFOAM bit for bit; NeoN's
 * own interpolation weights round differently. Boundary faces: (phir_b*alpha2_b)*alpha1_b.
 * interFoam passes phir = phic*nHatf.
 */
nnfvcc::SurfaceField<NeoN::scalar> linearCompressionFlux(
    RunTime& rt,
    const nnfvcc::VolumeField<NeoN::scalar>& alpha1,
    const nnfvcc::SurfaceField<NeoN::scalar>& phir
);

/* @brief linearCompressionFlux into an existing field. */
void linearCompressionFlux(
    RunTime& rt,
    const nnfvcc::VolumeField<NeoN::scalar>& alpha1,
    const nnfvcc::SurfaceField<NeoN::scalar>& phir,
    nnfvcc::SurfaceField<NeoN::scalar>& flux
);

/* @brief alphaPhi += linearCompressionFlux(alpha1, phic*nHatf), internal and boundary faces.
 *
 * @detail The same face arithmetic as forming phir = phic*nHatf, the compression flux and the
 * sum one after the other, in one kernel without the intermediate fields: vanLeerAlphaFlux
 * into alphaPhi followed by this gives alphaPhaseFlux.
 */
void addInterfaceCompressionFlux(
    RunTime& rt,
    const nnfvcc::VolumeField<NeoN::scalar>& alpha1,
    const nnfvcc::SurfaceField<NeoN::scalar>& phic,
    const nnfvcc::SurfaceField<NeoN::scalar>& nHatf,
    nnfvcc::SurfaceField<NeoN::scalar>& alphaPhi
);

/* @brief interFoam alphaEqn.H high-order phase flux alphaPhiUn.
 *
 * @detail vanLeerAlphaFlux(alpha1) + linearCompressionFlux(alpha1, phir) with
 * phir = interfaceCompressionVelocity(phi, cAlpha)*computeNHatf(alpha1): linear alpharScheme,
 * icAlpha == scAlpha == 0. No compression on (non-coupled) boundary faces. The same face
 * arithmetic as that composition, fused into one kernel without its temporary fields.
 */
nnfvcc::SurfaceField<NeoN::scalar> alphaPhaseFlux(
    RunTime& rt,
    const nnfvcc::VolumeField<NeoN::scalar>& alpha1,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi,
    double cAlpha
);

} // namespace NeoFOAM
