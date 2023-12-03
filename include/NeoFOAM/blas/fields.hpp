// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "primitives/scalar.hpp"
#include "primitives/vector.hpp"
#include "deviceField.hpp"

namespace NeoFOAM
{

using labelField = NeoFOAM::deviceField<int32_t>;

using scalarField = NeoFOAM::deviceField<scalar>;
using vectorField = NeoFOAM::deviceField<vector>;



} // namespace NeoFOAM