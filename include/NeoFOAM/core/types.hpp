// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <concepts>

#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/core/primitives/vector.hpp"

namespace NeoFOAM
{

template<class T>
concept ValueType = std::floating_point<T> || std::is_same_v<T, Vector>;

}
