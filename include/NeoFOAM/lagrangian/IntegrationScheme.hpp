// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include <string>

namespace NeoFOAM::lagrangian
{

// Wrapper method for integration scheme, either Euler or RK2 (Heun's method)

enum class IntegrationScheme
{
    Euler,      ///< 1st-order implicit Euler (as OpenFOAM)
    analytical, ///< exact for linear drag (placeholder) // exponential decay only work on the Re
                ///< inside the stoke regime
    RK2         ///< 2nd-order Heun's method
};

/* @brief Map a config string to an IntegrationScheme enum value. */
inline IntegrationScheme integrationSchemeFromName(const std::string& name)
{
    if (name == "Euler") return IntegrationScheme::Euler;
    if (name == "analytical")
        return IntegrationScheme::analytical; //  exact for linear drag (placeholder) exponential
                                              //  decay
    if (name == "RK2") return IntegrationScheme::RK2;

    NF_ERROR_EXIT(
        "Unknown particle integration scheme '" + name + "' (valid: Euler analytical RK2)"
    );

    return IntegrationScheme::Euler; // unreachable, silences compiler warning
}

} // namespace NeoFOAM::lagrangian
