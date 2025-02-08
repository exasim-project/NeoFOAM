// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors


namespace NeoFOAM
{
template<typename T>
struct one
{
    static const T value = 1;
};

template<typename T>
struct zero
{
    static const T value = 0;
};


}
