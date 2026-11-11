// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

namespace NeoFOAM::lagrangian
{

/* @brief Schiller-Naumann drag correlation for a rigid sphere.
 *
 *     Cd = (24 / Re) * (1 + 0.15 * Re^0.687)   for Re <= 1000
 *     Cd = 0.44                                for Re >  1000
 *
 * so Cd * Re = 24 * (1 + 0.15 * Re^0.687), or 0.44 * Re above Re = 1000.
 *
 * Widely used in Euler-Lagrange solvers. Note: OpenFOAM's "sphereDrag" is
 * the Putnam correlation (SphereDrag), not this one; the two give slightly
 * different terminal velocities.
 *
 * Callable on the GPU (KOKKOS_INLINE_FUNCTION).
 */
class SchillerNaumann
{
public:

    KOKKOS_INLINE_FUNCTION
    NeoN::scalar cdRe(NeoN::scalar re) const
    {
        if (re > 1000.0)
        {
            return 0.44 * re;
        }
        return 24.0 * (1.0 + 0.15 * Kokkos::pow(re, 0.687));
    }
};

} // namespace NeoFOAM::lagrangian
