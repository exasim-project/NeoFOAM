// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include "ParticleCloud.hpp"

#include <ostream>

namespace NeoFOAM::lagrangian
{

/* @brief Write the CSV header line: t,id,x,y,z,ux,uy,uz,d (and set 17 digits). */
void writeParticlesHeader(std::ostream& out);

/* @brief Append one row per particle at time t: t,id,x,y,z,ux,uy,uz,d.
 *
 * The particle data is copied to the host first (no-op copy on Serial/CPU).
 * Inactive particles are written too, at their last position.
 */
void writeParticles(std::ostream& out, NeoN::scalar t, const ParticleCloud& cloud);

} // namespace NeoFOAM::lagrangian
