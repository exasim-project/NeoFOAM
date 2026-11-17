// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "ForceContext.hpp"

namespace NeoFOAM::lagrangian
{

/* @brief Aerodynamic drag force on a rigid sphere.
 *
 *     F_D = 0.5 * Cd * rho_f * A_p * |slip| * slip
 *
 * where:
 *     slip  = uf - up           (relative velocity)
 *     A_p   = pi * d^2 / 4      (projected area of the sphere)
 *     Cd    = drag coefficient
 *
 * With Re = rho_f * |slip| * d / mu_f this is the same as
 *
 *     F_D = (pi / 8) * mu_f * d * (Cd * Re) * slip
 *
 * which is what is computed: Cd * Re (from the DragModel, passed in ctx) is
 * finite at Re = 0, Cd alone is not.
 *
 * The drag force always points in the direction of the slip vector:
 * it accelerates the particle toward the fluid velocity.
 *
 * Callable on the GPU (KOKKOS_INLINE_FUNCTION).
 */

class DragForce
{
public:

    // Method to compute the drag force on a particle given its velocity, diameter, and the force
    // context (which includes fluid properties and drag coefficient). The force is calculated based
    // on the slip velocity between the fluid and the particle.
    KOKKOS_INLINE_FUNCTION
    NeoN::Vec3
    compute(const NeoN::Vec3& up, NeoN::scalar d, NeoN::scalar, const ForceContext& ctx) const
    {
        // Slip velocity: fluid minus particle - relative velocity of the particle with respect to
        // the fluid
        const NeoN::Vec3 slip = ctx.uf - up;

        // F = (pi / 8) * mu * d * CdRe * slip
        const NeoN::scalar k = M_PI / 8.0 * ctx.muF * d * ctx.cdRe;

        return slip * k;
    }


    // Method to compute the drag coefficient: F_D = Sp * slip, Sp = (pi / 8) * mu * d * CdRe.
    KOKKOS_INLINE_FUNCTION
    NeoN::scalar sp(NeoN::scalar d, NeoN::scalar, const ForceContext& ctx) const
    {
        return M_PI / 8.0 * ctx.muF * d * ctx.cdRe;
    }
};

} // namespace NeoFOAM::lagrangian
