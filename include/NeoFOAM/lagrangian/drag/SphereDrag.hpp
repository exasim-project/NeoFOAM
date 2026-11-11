// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

namespace NeoFOAM::lagrangian
{

/* @brief Putnam drag correlation for a rigid sphere, as OpenFOAM's
 * "sphereDrag" (SphereDragForce::CdRe).
 *
 *     Cd * Re = 24 * (1 + Re^(2/3) / 6)   for Re <= 1000
 *     Cd * Re = 0.424 * Re                for Re >  1000 (Newton regime, Cd = 0.424)
 *
 * Recovers Stokes drag (Cd * Re = 24) as Re -> 0.
 *
 * Callable on the GPU (KOKKOS_INLINE_FUNCTION).
 */
class SphereDrag
{
public:

    KOKKOS_INLINE_FUNCTION
    NeoN::scalar cdRe(NeoN::scalar re) const
    {
        if (re > 1000.0)
        {
            return 0.424 * re;
        }
        return 24.0 * (1.0 + Kokkos::pow(re, 2.0 / 3.0) / 6.0);
    }
};

} // namespace NeoFOAM::lagrangian
