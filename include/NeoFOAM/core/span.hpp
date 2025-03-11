// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
#pragma once

#include <span>

namespace NeoFOAM
{

/* @class Span
 *
 * @brief A wrapper class for std::span which allows to check whether the index access is in range
 * The Span can be initialized like a regular std::span or from an existing std::span
 *
 * @ingroup core
 *
 */
template<typename ValueType>
class Span : public std::span<ValueType>
{
public:

    /* A flag to control whether the program should terminate on invalid memory access or throw.
     * Kokkos prefers to terminate, but for testing purpose throwing is preferred
     */
    bool abort = true;

    using std::span<ValueType>::span; // Inherit constructors from std::span

    /* Constructor from existing std::span
     */
    Span(std::span<ValueType> in) : Span(in.begin(), in.end()) {}

    constexpr ValueType& operator[](std::size_t index) const
    {
#ifdef NF_DEBUG
        if (index >= this->size())
        {
            std::string msg("Index is out of range. Index: " + std::to_string(index));
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
