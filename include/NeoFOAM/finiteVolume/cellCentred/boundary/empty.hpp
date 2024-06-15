// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once
#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType, typename BaseType>
class Empty : public BaseType<ValueType>
{
public:

    using BaseType<ValueType>::BaseType;

    virtual void
    correctBoundaryConditions(BoundaryFields<ValueType>& bfield, Field<ValueType>& internalField);
};

using EmptySurfaceScalarBoundary = Empty<scalar, SurfaceBoundaryBase<scalar>>;
using EmptySurfaceVolumeBoundary = Empty<scalar, VolumeBoundaryBase<scalar>>;

};
