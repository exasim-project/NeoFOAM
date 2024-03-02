// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/primitives/scalar.hpp"
#include "NeoFOAM/primitives/vector.hpp"
#include "NeoFOAM/primitives/label.hpp"
#include "NeoFOAM/fields/Field.hpp"
#include "NeoFOAM/fields/FieldOperations.hpp"

namespace NeoFOAM
{

using labelField = NeoFOAM::Field<label>;

using scalarField = NeoFOAM::Field<scalar>;
using vectorField = NeoFOAM::Field<vector>;


} // namespace NeoFOAM