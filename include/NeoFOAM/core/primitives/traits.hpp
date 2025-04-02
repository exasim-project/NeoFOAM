// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <Kokkos_Core.hpp> // IWYU pragma: keep

namespace NeoFOAM
{

template<typename T>
KOKKOS_INLINE_FUNCTION T one()
{
    const T value;
    return value;
};

template<typename T>
KOKKOS_INLINE_FUNCTION T zero()
{
    const T value;
    return value;
};


#define NF_FOR_ALL_INDEX_TYPES(FUNC)                                                               \
    FUNC(unsigned char);                                                                           \
    FUNC(label);                                                                                   \
    FUNC(localIdx);                                                                                \
    FUNC(globalIdx)

#define NF_FOR_ALL_VALUE_TYPES(FUNC)                                                               \
    FUNC(scalar);                                                                                  \
    FUNC(Vector)

}
