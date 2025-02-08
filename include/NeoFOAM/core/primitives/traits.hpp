// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

namespace NeoFOAM
{

template<typename T>
struct one
{
    static const T value;
};

template<typename T>
struct zero
{
    static const T value;
};

}
