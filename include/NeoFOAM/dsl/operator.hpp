// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

namespace NeoFOAM::dsl
{

class Operator
{
public:

    enum class Type
    {
        Implicit,
        Explicit
    };
};

} // namespace NeoFOAM::dsl
