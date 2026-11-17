// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "NeoN/NeoN.hpp"

namespace NeoFOAM
{

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

/* @brief The interface curvature K = -fvc::div(nHatf) of interfaceProperties::calculateK.
 *
 * @detail fvc::surfaceIntegrate's sums (owner +, neighbour -, boundary owner +, then / V),
 * negated; boundary values are the owner-cell values (extrapolatedCalculated). Serial sums in
 * OpenFOAM's face order; the parallel backends add atomically.
 */
void interfaceCurvature(
    const nnfvcc::SurfaceField<NeoN::scalar>& nHatf,
    nnfvcc::VolumeField<NeoN::scalar>& K
);

} // namespace NeoFOAM
