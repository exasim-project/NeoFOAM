// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/auxiliary/convert.hpp"

#include "NeoN/NeoN.hpp"

#include <catch2/catch_approx.hpp>


struct ApproxScalar
{
    Foam::scalar margin;
    bool operator()(double rhs, double lhs) const
    {
        return Catch::Approx(rhs).margin(margin) == lhs;
    }
};

struct ApproxVector
{
    NeoN::Vec3 margin;

    ApproxVector(NeoN::Vec3 v)
        : margin(v)
    {}
    ApproxVector(NeoN::scalar v)
        : margin({v, v, v})
    {}

    bool operator()(NeoN::Vec3 rhs, Foam::vector lhs) const
    {
        NeoN::Vec3 diff(rhs[0] - lhs[0], rhs[1] - lhs[1], rhs[2] - lhs[2]);

        return Catch::Approx(0).margin(margin[0]) == diff[0]
            && Catch::Approx(0).margin(margin[1]) == diff[1]
            && Catch::Approx(0).margin(margin[2]) == diff[2];
    }
};
