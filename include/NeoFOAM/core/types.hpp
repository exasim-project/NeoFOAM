// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <concepts>

#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/core/primitives/vector.hpp"

namespace NeoFOAM
{

/**
 * Concept for types that can be stored by a Field.
 */
template<class T>
concept StorageType = std::is_standard_layout_v<T>;

/**
 * Concept for types that can be used in computations, i.e. floating point types or a 3D vector.
 */
template<class T>
concept ValueType = std::floating_point<T> || std::is_same_v<T, Vector>;

}
