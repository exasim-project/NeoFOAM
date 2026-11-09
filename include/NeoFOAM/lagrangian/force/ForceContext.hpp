// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

namespace NeoFOAM::lagrangian
{

/* @brief Context passed to force models.
 *
 * Contains everything a force model needs to know about the fluid
 * at the particle's current position, plus Cd * Re
 * (which is computed once per particle per step by the solver).
 */
struct ForceContext
{
    NeoN::Vec3 uf;     ///< fluid velocity at particle [m/s]
    NeoN::scalar cdRe; ///< drag coefficient times Reynolds number [-] (precomputed by solver)
    NeoN::scalar rhoF; ///< fluid density [kg/m^3]
    NeoN::scalar muF;  ///< fluid dynamic viscosity [Pa s]
};

} // namespace NeoFOAM::lagrangian
