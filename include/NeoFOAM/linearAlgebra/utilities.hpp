// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

namespace NeoFOAM
{

struct Dim
{
    size_t numRows;
    size_t numCols;
};


template<typename IndexType>
struct MatrixCoordinate
{
    IndexType row;
    IndexType col;
};

template<typename ValueType, typename IndexType>
struct MatrixEntry
{
    IndexType row;
    IndexType col;
    ValueType value;
};


struct CompatibleWithAnyExecutor
{
    static Executor getCompatibleExecutor(const Executor& exec) { return exec; }
};


}
