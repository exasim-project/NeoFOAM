// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/primitives/scalar.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "Kokkos_Core.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{
struct fixedValueBCKernel
{
    scalar uniformValue_;
    int start_;
    int end_;

    void operator()(const GPUExecutor& exec, boundaryFields<scalar>& bField);

    void operator()(const ompExecutor& exec, boundaryFields<scalar>& bField);

    void operator()(const CPUExecutor& exec, boundaryFields<scalar>& bField);
};

class fvccScalarFixedValueBoundaryField : public fvccBoundaryField<scalar>
{
public:

    fvccScalarFixedValueBoundaryField(int start, int end, scalar uniformValue);

    void correctBoundaryConditions(boundaryFields<scalar>& field);

private:

    scalar uniformValue_;
};
};