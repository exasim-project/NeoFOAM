// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "ForceContext.hpp"

namespace NeoFOAM::lagrangian
{

/* @brief Gravity + buoyancy force on a rigid sphere.
 *
 *     F_G = (rho_p - rho_f) * V_p * g
 *
 * where:
 *     V_p   = pi * d^3 / 6       (volume of the sphere)
 *     g     = gravity vector     (from constant/g, e.g. (0, 0, -9.81) m/s^2)
 *     rho_p = particle density
 *     rho_f = fluid density      (buoyancy correction)
 *
 * If rho_p > rho_f, the net force points in the direction of g
 * (the particle sinks / falls).
 *
 * If rho_p < rho_f, the net force points opposite to g
 * (the particle rises, e.g. a bubble).
 *
 * Callable on the GPU (KOKKOS_INLINE_FUNCTION).
 */
class GravityForce
{
public:

    NeoN::Vec3 g = {0.0, 0.0, 0.0}; ///< gravity vector [m/s^2], from constant/g

    /* @brief Compute gravity + buoyancy force on one particle. */
    KOKKOS_INLINE_FUNCTION
    NeoN::Vec3
    compute(const NeoN::Vec3&, NeoN::scalar d, NeoN::scalar rhoP, const ForceContext& ctx) const
    {
        // Volume of the sphere
        const NeoN::scalar vol = M_PI * d * d * d / 6.0;

        // Net density difference (particle minus fluid)
        const NeoN::scalar drho = rhoP - ctx.rhoF;

        // F = (rho_p - rho_f) * V * g
        return g * (drho * vol);
    }

    /* @brief Gravity does not depend on the slip: fully explicit, Sp = 0. */
    KOKKOS_INLINE_FUNCTION
    NeoN::scalar sp(NeoN::scalar, NeoN::scalar, const ForceContext&) const { return 0.0; }
};

} // namespace NeoFOAM::lagrangian
