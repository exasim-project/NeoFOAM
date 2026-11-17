// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

namespace NeoFOAM::lagrangian
{

/* @brief Stokes drag for a rigid sphere.
 *
 * Valid in the creeping-flow regime, Re_p < 1.
 *
 *     Cd = 24 / Re   ->   Cd * Re = 24
 *
 * This is the exact analytical solution for a sphere in Stokes flow.
 * For higher Re_p, use SphereDrag or SchillerNaumann instead.
 *
 * Callable on the GPU (KOKKOS_INLINE_FUNCTION).
 */
class StokesDrag
{
public:

    KOKKOS_INLINE_FUNCTION
    NeoN::scalar cdRe(NeoN::scalar) const { return 24.0; }
};

} // namespace NeoFOAM::lagrangian
