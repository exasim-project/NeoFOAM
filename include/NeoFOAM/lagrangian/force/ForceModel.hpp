// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include "DragForce.hpp"
#include "ForceContext.hpp"
#include "GravityForce.hpp"

namespace NeoFOAM::lagrangian
{

/* @brief The set of forces acting on the particles of the cloud.
 *
 * A force model answers ONE question:
 *     what force acts on this particle right now?
 *
 * Different models represent different physical effects:
 *     DragForce       —  drag
 *     GravityForce    — gravity + buoyancy
 *     SaffmanLift     — shear-induced lift // Not yet implemented or consider
 *     VirtualMass     — added mass        // Not yet implemented or consider
 *     PressureGradient — pressure gradient force // Not yet implemented or consider
 *
 * Fields are passed individually (SoA-friendly) instead of a
 * Particle& so that the same interface works on CPU and GPU.
 *
 * GPU: a list of virtual ForceModel pointers cannot be called inside a GPU
 * kernel, so every force is a member here, switched on or off by a flag.
 * ForceModels is a small value type, copied into the kernel.
 *
 * To add a force: write a class with KOKKOS_INLINE_FUNCTION compute(...) and
 * sp(...), add a flag and a member below, and add it to compute() and sp().
 */
class ForceModels
{
public:

    bool useDrag = false;    ///< subModels/particleForces has a drag model
    bool useGravity = false; ///< subModels/particleForces has gravity

    DragForce drag;
    GravityForce gravity;

    /* @brief Total force on one particle [N].
     *
     * @param up      particle velocity [m/s]
     * @param d       particle diameter [m]
     * @param rhoP    particle density [kg/m^3]
     * @param ctx     fluid context at the particle's position
     */

    KOKKOS_INLINE_FUNCTION
    NeoN::Vec3
    compute(const NeoN::Vec3& up, NeoN::scalar d, NeoN::scalar rhoP, const ForceContext& ctx) const
    {
        NeoN::Vec3 force = {0.0, 0.0, 0.0};
        if (useDrag)
        {
            force += drag.compute(up, d, rhoP, ctx);
        }
        if (useGravity)
        {
            force += gravity.compute(up, d, rhoP, ctx);
        }
        return force;
    }

    /* @brief Total implicit coefficient Sp [kg/s], as OpenFOAM's forceSuSp.
     *
     * The part of the force that is linear in the slip: F = ... + Sp * (uf - up).
     * Used by the implicit Euler integrator.
     */
    KOKKOS_INLINE_FUNCTION
    NeoN::scalar sp(NeoN::scalar d, NeoN::scalar rhoP, const ForceContext& ctx) const
    {
        NeoN::scalar total = 0.0;
        if (useDrag)
        {
            total += drag.sp(d, rhoP, ctx);
        }
        if (useGravity)
        {
            total += gravity.sp(d, rhoP, ctx);
        }
        return total;
    }

    /* @brief Number of active forces (host, for the log). */
    int size() const { return int(useDrag) + int(useGravity); }
};

} // namespace NeoFOAM::lagrangian
