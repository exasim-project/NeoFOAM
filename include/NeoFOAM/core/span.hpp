// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <span>

#include "NeoFOAM/core/primitives/label.hpp"


namespace NeoFOAM
{


template<typename ValueType>
class Span
{
public:

    template<
        typename... Args,
        typename = std::enable_if_t<std::is_constructible_v<std::span<ValueType>, Args...>>>
    constexpr explicit Span(Args&&... args) : span_(std::forward<Args>(args)...)
    {}

    constexpr auto begin() const { return span_.begin(); }

    constexpr auto end() const { return span_.end(); }

    template<typename IndexType>
    constexpr ValueType& operator[](IndexType i) const
    {
        return span_[static_cast<std::size_t>(i)];
    }

    constexpr size_t size() const { return static_cast<size_t>(span_.size()); }

    constexpr explicit(false) operator std::span<ValueType>() const { return span_; }

private:

    std::span<ValueType> span_;
};


}
