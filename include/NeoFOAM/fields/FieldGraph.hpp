// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "NeoFOAM/primitives/label.hpp"
#include "NeoFOAM/primitives/scalar.hpp"
#include "Field.hpp"

#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{
// enum class boundaryType used in a kokkos view to categorize boundary
// types


template<typename T>
class FieldGraph
{


public:

    FieldGraph(const FieldGraph<T>& rhs)
        : exec_(rhs.exec_),
          value_(rhs.value_),
          offset_(rhs.offset_)
    {
    }


    FieldGraph(const executor& exec, int nBoundaryFaces, int nBoundaries)
        : exec_(exec),
          value_(exec, nBoundaryFaces),
          offset_(exec, nBoundaries + 1)
    {
    }


    const NeoFOAM::Field<T>& value() const
    {
        return value_;
    }

    NeoFOAM::Field<T>& value()
    {
        return value_;
    }

    const NeoFOAM::Field<localIdx>& offset() const
    {
        return offset_;
    }

    const executor& exec()
    {
        return exec_;
    }

private:

    executor exec_;
    NeoFOAM::Field<T> value_;              ///< The view storing the computed values from the boundary condition
    NeoFOAM::Field<localIdx> offset_;      ///< The view storing the offsets
};

} // namespace NeoFOAM
