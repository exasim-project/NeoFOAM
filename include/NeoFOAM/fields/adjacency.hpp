// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "primitives/label.hpp"
#include "deviceAdjacency.hpp"

namespace NeoFOAM
{

template<bool directed>
using localAdjacency = NeoFOAM::deviceAdjacency<localIdx, directed>;

template<bool directed>
using globalAdjacency = NeoFOAM::deviceAdjacency<globalIdx, directed>;

} // namespace NeoFOAM