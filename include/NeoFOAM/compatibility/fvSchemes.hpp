// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoN/core/dictionary.hpp"


namespace NeoFOAM
{

void updateDdtSchemes(NeoN::Dictionary& solverDict);

NeoN::Dictionary mapFvSchemes(const NeoN::Dictionary& schemesDict);

} // namespace NeoFOAM
