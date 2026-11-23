// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "NeoN/NeoN.hpp"

namespace NeoFOAM
{

namespace
{

using NeoN::scalar;

// surfaceInterpolationScheme::interpolate writes the face value as w*(P - N) + N; any other
// arrangement of the same weights rounds differently.
KOKKOS_INLINE_FUNCTION scalar faceValue(scalar w, scalar valueP, scalar valueN)
{
    return w * (valueP - valueN) + valueN;
}

} // namespace

} // namespace NeoFOAM
