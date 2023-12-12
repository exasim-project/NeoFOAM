// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "primitives/label.hpp"
#include "deviceAdjacency.hpp"

namespace NeoFOAM
{

using localAdjacency = NeoFOAM::deviceAdjacency<localIdx>;
using globalAdjacency = NeoFOAM::deviceAdjacency<globalIdx>;

} // namespace NeoFOAM