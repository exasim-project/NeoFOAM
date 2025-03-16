// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

namespace NeoFOAM
{

template<typename T>
T one()
{
    const T value;
    return value;
};

template<typename T>
T zero()
{
    const T value;
    return value;
};

}
