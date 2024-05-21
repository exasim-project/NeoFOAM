// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/Field.hpp"
#include "NeoFOAM/fields/FieldOperations.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/core/primitives/vector.hpp"


namespace NeoFOAM
{

using labelField = NeoFOAM::Field<label>;
using scalarField = NeoFOAM::Field<scalar>;
using vectorField = NeoFOAM::Field<Vector>;

} // namespace NeoFOAM
