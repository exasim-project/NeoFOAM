// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "Integrator.hpp"

namespace NeoFOAM::lagrangian
{

// Just the Intergator for the RK2 scheme, use by evolve() in ParticleSolver. EulerIntegrator is in
// EulerIntegrator.hpp
class RK2Integrator
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
        // keep the old state: both corrector formulas need u and x of the start of the step
        const NeoN::Vec3 u0 = u;
        const NeoN::Vec3 x0 = x;

        // ─── Predictor ───
        const NeoN::Vec3 a1 = accel(u0, x0).a;
        const NeoN::Vec3 uStar = u0 + a1 * dt;
        const NeoN::Vec3 xStar = x0 + u0 * dt;

        // ─── Corrector ───
        const NeoN::Vec3 a2 = accel(uStar, xStar).a;

        u = u0 + (a1 + a2) * (0.5 * dt);
        x = x0 + (u0 + uStar) * (0.5 * dt);
    }
};

} // namespace NeoFOAM::lagrangian
