// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "Integrator.hpp"

namespace NeoFOAM::lagrangian
{

// Just the Intergator for the Euler scheme, use by evolve() in ParticleSolver. RK2Integrator is in
// RK2Integrator.hpp
class EulerIntegrator
{
public:

    /* @brief Advance one particle by dt.
     *
     * @param u      particle velocity (updated in place)
     * @param x      particle position (updated in place)
     * @param accel  callback that computes acceleration (see Integrator.hpp)
     * @param dt     timestep [s]
     */
    template<typename AccelFunc>
    KOKKOS_INLINE_FUNCTION void
    step(NeoN::Vec3& u, NeoN::Vec3& x, const AccelFunc& accel, NeoN::scalar dt) const
    {
        // Evaluate acceleration and implicit coefficient at the current state
        const Acceleration acc = accel(u, x);

        // Position with the old velocity, then implicit velocity update
        x += u * dt;
        u += acc.a * (dt / (1.0 + acc.bp * dt));
    }
};

} // namespace NeoFOAM::lagrangian
