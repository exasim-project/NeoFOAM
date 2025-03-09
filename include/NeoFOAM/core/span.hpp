// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
#pragma once

#include <span>

namespace NeoFOAM
{

template<typename T>
class Span : public std::span<T>
{
public:

    using std::span<T>::span; // Inherit constructors from std::span

    constexpr T& operator[](std::size_t index) const
    {
#ifdef DEBUG
        if (index >= this->size())
        {
            Kokkos::abort("Index out of range in Span.\n");
        }
#endif
        return std::span<T>::operator[](index);
    }
};


} // namespace NeoFOAM
