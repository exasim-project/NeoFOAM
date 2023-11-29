// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "deviceField.hpp"

namespace NeoFOAM
{

using labelField = NeoFOAM::deviceField<int32_t, 1>;

using scalarField = NeoFOAM::deviceField<double, 1>;
using vectorField = NeoFOAM::deviceField<double, 3>;



} // namespace NeoFOAM