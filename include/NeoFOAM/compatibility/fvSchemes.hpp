// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/tokenList.hpp"


namespace NeoFOAM
{

void updateDdtSchemes(NeoN::Dictionary& solverDict);

NeoN::Dictionary mapFvSchemes(const NeoN::Dictionary& schemesDict);

} // namespace NeoFOAM
