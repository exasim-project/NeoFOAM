// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once
#include "NeoN/NeoN.hpp"

namespace NeoFOAM::lagrangian
{

// Struct to hold acceleration and implicit coefficient, returned by the acceleration callback in
// Integrator.hpp

struct Acceleration
{
    NeoN::Vec3 a;
    NeoN::scalar bp;
};

} // namespace NeoFOAM::lagrangian
