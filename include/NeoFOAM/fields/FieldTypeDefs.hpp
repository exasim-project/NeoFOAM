// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/Field.hpp"
#include "NeoFOAM/primitives/scalar.hpp"
#include "NeoFOAM/primitives/vector.hpp"

namespace NeoFOAM
{

using labelField = NeoFOAM::Field<label>;
using scalarField = NeoFOAM::Field<scalar>;
using vectorField = NeoFOAM::Field<Vector>;

} // namespace NeoFOAM
