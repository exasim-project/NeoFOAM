// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once
#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

namespace NeoFOAM
{
struct ZeroGradientBCKernel
{
    const UnstructuredMesh& mesh_;
    int patchID_;
    int start_;
    int end_;

    ZeroGradientBCKernel(const UnstructuredMesh& mesh, int patchID, int start, int end)
        : mesh_(mesh), patchID_(patchID), start_(start), end_(end)
    {
        // TODO add asserts end > start
    }

    void operator()(
        const GPUExecutor& exec, BoundaryFields<scalar>& bField, const Field<scalar>& internalField
    );

    void operator()(
        const OMPExecutor& exec, BoundaryFields<scalar>& bField, const Field<scalar>& internalField
    );

    void operator()(
        const CPUExecutor& exec, BoundaryFields<scalar>& bField, const Field<scalar>& internalField
    );
};

class fvccScalarZeroGradientBoundaryField : public fvccBoundaryField<scalar>
{
public:

    fvccScalarZeroGradientBoundaryField(const UnstructuredMesh& mesh, int patchID);

    virtual void
    correctBoundaryConditions(BoundaryFields<scalar>& bfield, const Field<scalar>& internalField);

private:
};
};
