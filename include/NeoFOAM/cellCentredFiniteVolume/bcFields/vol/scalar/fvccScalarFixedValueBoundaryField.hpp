// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once
#include "Kokkos_Core.hpp"

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "NeoFOAM/mesh/unstructured/UnstructuredMesh.hpp"

namespace NeoFOAM
{
struct fixedValueBCKernel
{
    const UnstructuredMesh& mesh_;
    int patchID_;
    int start_;
    int end_;
    scalar uniformValue_;

    fixedValueBCKernel(
        const UnstructuredMesh& mesh, int patchID, int start, int end, scalar uniformValue
    )
        : mesh_(mesh), patchID_(patchID), start_(start), end_(end), uniformValue_(uniformValue)
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

class fvccScalarFixedValueBoundaryField : public fvccBoundaryField<scalar>
{
public:

    fvccScalarFixedValueBoundaryField(
        const UnstructuredMesh& mesh, int patchID, scalar uniformValue
    );

    virtual void
    correctBoundaryConditions(BoundaryFields<scalar>& bfield, const Field<scalar>& internalField);

private:

    scalar uniformValue_;
};
};
