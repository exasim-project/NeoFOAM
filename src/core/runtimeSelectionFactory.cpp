// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/core/runtimeSelectionFactory.hpp"

std::unordered_map<std::string, NeoFOAM::Dictionary>& NeoFOAM::singletonLookupTable()
{
    static std::unordered_map<std::string, NeoFOAM::Dictionary> table;
    return table;
}
