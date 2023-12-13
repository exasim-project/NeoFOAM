// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <array>
#include <Kokkos_Core.hpp>

#include "scalar.hpp"

namespace NeoFOAM
{
    class vector
    {
    public:
        KOKKOS_INLINE_FUNCTION
        vector() {}

        KOKKOS_INLINE_FUNCTION
        vector(scalar x, scalar y, scalar z) : cmpts_({x, y, z}) {}

        KOKKOS_INLINE_FUNCTION
        std::array<scalar, 3>::const_iterator begin() const noexcept
        {
            return cmpts_.begin();
        }

        KOKKOS_INLINE_FUNCTION
        std::array<scalar, 3>::const_iterator end() const noexcept
        {
            return cmpts_.end();
        }

        KOKKOS_INLINE_FUNCTION
        scalar& operator()(const int i)
        {
            return cmpts_[i];
        }

        KOKKOS_INLINE_FUNCTION
        scalar operator()(const int i) const
        {
            return cmpts_[i];
        }

        KOKKOS_INLINE_FUNCTION
        vector& operator+=(const vector& other)
        {
            for(auto i = 0; i < cmpts_.size(); ++i) cmpts_[i] += other.cmpts_[i]; 
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        vector& operator-=(const vector& other)
        {
            for(auto i = 0; i < cmpts_.size(); ++i) cmpts_[i] -= other.cmpts_[i]; 
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        vector& operator*=(const scalar &sclr)
        {
            for(auto& comp : cmpts_) comp *= sclr; 
            return *this;
        }

    private:
        std::array<scalar, 3> cmpts_ = {0.0, 0.0, 0.0};
    };


KOKKOS_INLINE_FUNCTION
bool operator==(const vector& lhs, const vector& rhs)
{
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

KOKKOS_INLINE_FUNCTION
vector operator+(vector lhs, const vector& rhs)
{
    lhs += rhs;
    return lhs;
}

KOKKOS_INLINE_FUNCTION
vector operator-(vector lhs, const vector& rhs)
{
    lhs -= rhs;
    return lhs;
}

KOKKOS_INLINE_FUNCTION
vector operator*(const scalar &sclr, vector rhs)
{   
    rhs *= sclr;
    return rhs;
}
} // namespace NeoFOAM
