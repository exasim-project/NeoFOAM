// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <utility>

#include "NeoN/NeoN.hpp"

namespace NeoFOAM
{

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

/* @brief Flow and interface Courant numbers of advectorVoF's setDeltaT.H, as {CoNum, alphaCoNum}.
 *
 * @detail sumPhi = fvc::surfaceSum(mag(phi)), CoNum = 0.5*gMax(sumPhi/V)*deltaT
 * (CourantNo.H) and alphaCoNum = 0.5*gMax(pos(alpha - 0.01)*pos(0.99 - alpha)*sumPhi/V)*deltaT
 * (advectorVoF's alphaCourantNo.H, with strict inequalities). Each cell sums its faces in
 * OpenFOAM's order (internal faces by index, then boundary faces), so the result is
 * bitwise OpenFOAM's on every executor. The cell-to-face map is built once per mesh and cached
 * in its stencil database. Only the internal field of @p alpha is read.
 */
std::pair<NeoN::scalar, NeoN::scalar> courantNumbers(
    const nnfvcc::VolumeField<NeoN::scalar>& alpha,
    const nnfvcc::SurfaceField<NeoN::scalar>& phi,
    NeoN::scalar deltaT
);

} // namespace NeoFOAM
