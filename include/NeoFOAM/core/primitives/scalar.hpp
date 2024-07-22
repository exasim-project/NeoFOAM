// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

// TODO this needs to be implemented in the corresponding cmake file
namespace NeoFOAM
{
#ifdef NEOFOAM_DP_SCALAR
using scalar = double;
#else
using scalar = float;
#endif
} // namespace NeoFOAM
