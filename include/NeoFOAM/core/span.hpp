// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
#pragma once

#include <span>

namespace NeoFOAM
{

template<typename ValueType>
class Span : public std::span<ValueType>
{
public:

    bool abort = true;

    using std::span<ValueType>::span; // Inherit constructors from std::span

    Span(std::span<ValueType> in) : Span(in.begin(), in.end()) {}

    constexpr ValueType& operator[](std::size_t index) const
    {
#ifdef NF_DEBUG
        if (index >= this->size())
        {
            std::string msg;
            msg += "Index is out of range. Index: " + std::to_string(index);
            if (abort)
            {
                Kokkos::abort(msg.c_str());
            }
            else
            {
                throw std::invalid_argument(msg);
            }
        }
#endif
        return std::span<ValueType>::operator[](index);
    }
};


} // namespace NeoFOAM
