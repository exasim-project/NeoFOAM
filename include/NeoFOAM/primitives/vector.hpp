// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "scalar.hpp"
#include <Kokkos_Core.hpp>

namespace NeoFOAM
{


/**
 * @class Vector
 * @brief A class for the representation of a 3D Vector
 */
class Vector
{
public:

    KOKKOS_INLINE_FUNCTION
    Vector()
    {
        cmpts_[0] = 0.0;
        cmpts_[1] = 0.0;
        cmpts_[2] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    Vector(scalar x, scalar y, scalar z)
    {
        cmpts_[0] = x;
        cmpts_[1] = y;
        cmpts_[2] = z;
    }

    KOKKOS_INLINE_FUNCTION
    scalar& operator()(const int i) { return cmpts_[i]; }

    KOKKOS_INLINE_FUNCTION
    scalar operator()(const int i) const { return cmpts_[i]; }

    KOKKOS_INLINE_FUNCTION
    bool operator==(const Vector& rhs) const
    {
        return cmpts_[0] == rhs(0) && cmpts_[1] == rhs(1) && cmpts_[2] == rhs(2);
    }

    KOKKOS_INLINE_FUNCTION
    Vector operator+(const Vector& rhs)
    {
        return Vector(cmpts_[0] + rhs(0), cmpts_[1] + rhs(1), cmpts_[2] + rhs(2));
    }

    KOKKOS_INLINE_FUNCTION
    Vector operator-(const Vector& rhs)
    {
        return Vector(cmpts_[0] - rhs(0), cmpts_[1] - rhs(1), cmpts_[2] - rhs(2));
    }

    KOKKOS_INLINE_FUNCTION
    Vector operator*(const scalar& rhs)
    {
        return Vector(cmpts_[0] * rhs, cmpts_[1] * rhs, cmpts_[2] * rhs);
    }

private:

    scalar cmpts_[3];
};
} // namespace NeoFOAM
