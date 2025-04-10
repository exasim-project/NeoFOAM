// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#pragma once

#include <Kokkos_Core.hpp> // IWYU pragma: keep

namespace NeoN
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

}
