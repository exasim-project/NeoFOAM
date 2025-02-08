// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/core/primitives/traits.hpp"

// TODO this needs to be implemented in the corresponding cmake file
namespace NeoFOAM
{
#ifdef NEOFOAM_DP_SCALAR
typedef double scalar;
#else
typedef float scalar;
#endif

constexpr scalar ROOTVSMALL = 1e-18;

// traits for vector
template<>
struct one<scalar>
{
    static const inline scalar value = 1.0;
};


template<>
struct zero<scalar>
{
    static const inline scalar value = 0.0;
};

} // namespace NeoFOAM
