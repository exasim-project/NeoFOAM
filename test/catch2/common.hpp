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
    Foam::scalar margin;
    bool operator()(NeoN::Vec3 rhs, Foam::vector lhs) const
    {
        NeoN::Vec3 diff(rhs[0] - lhs[0], rhs[1] - lhs[1], rhs[2] - lhs[2]);

        return Catch::Approx(0).margin(margin) == mag(diff);
    }
};
