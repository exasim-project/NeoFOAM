// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "Field.hpp"
#include "primitives/scalar.hpp"
#include "primitives/vector.hpp"

namespace NeoFOAM {

using labelField = NeoFOAM::Field<label>;
using scalarField = NeoFOAM::Field<scalar>;
using vectorField = NeoFOAM::Field<vector>;

} // namespace NeoFOAM
