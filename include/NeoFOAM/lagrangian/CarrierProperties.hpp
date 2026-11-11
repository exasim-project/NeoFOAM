// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

namespace NeoFOAM::lagrangian
{

// Struct to hold carrier properties, density and  viscosity

struct CarrierProperties
{
    NeoN::scalar rho; ///< carrier density [kg/m^3]
    NeoN::scalar mu;  ///< carrier dynamic viscosity [Pa s]
};

} // namespace NeoFOAM::lagrangian
