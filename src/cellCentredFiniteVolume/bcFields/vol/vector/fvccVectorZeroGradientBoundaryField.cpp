// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorZeroGradientBoundaryField.hpp"

namespace NeoFOAM
{
fvccVectorZeroGradientBoundaryField::fvccVectorZeroGradientBoundaryField(const unstructuredMesh& mesh, int patchID)
    : fvccBoundaryField<Vector>(mesh, patchID)
{
}

void fvccVectorZeroGradientBoundaryField::correctBoundaryConditions(boundaryFields<Vector>& bfield, const Field<Vector>& internalField)
{
    ZeroGradientVectorBCKernel kernel_(mesh_, patchID_, start_, end_);
    std::visit([&](const auto& exec)
               { kernel_(exec, bfield, internalField); },
               bfield.exec());
}


void ZeroGradientVectorBCKernel::operator()(const GPUExecutor& exec, boundaryFields<Vector>& bField, const Field<Vector>& internalField)
{
    using executor = typename GPUExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refGrad = bField.refGrad().field();
    auto faceCells = mesh_.boundaryMesh().faceCells(patchID_);
    const auto iField = internalField.field();
    Kokkos::parallel_for(
        "fvccVectorZeroGradientBoundaryField", Kokkos::RangePolicy<executor>(start_, end_), KOKKOS_LAMBDA(const int i) {
            s_value[i] = iField[faceCells[i]];
            s_refGrad[i] = Vector(0, 0, 0);
        }
    );
}

void ZeroGradientVectorBCKernel::operator()(const OMPExecutor& exec, boundaryFields<Vector>& bField, const Field<Vector>& internalField)
{
    using executor = typename OMPExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refGrad = bField.refGrad().field();
    auto faceCells = mesh_.boundaryMesh().faceCells(patchID_);
    const auto iField = internalField.field();
    Kokkos::parallel_for(
        "fvccVectorZeroGradientBoundaryField", Kokkos::RangePolicy<executor>(start_, end_), KOKKOS_LAMBDA(const int i) {
            s_value[i] = iField[faceCells[i]];
            s_refGrad[i] = Vector(0, 0, 0);
        }
    );
}

void ZeroGradientVectorBCKernel::operator()(const CPUExecutor& exec, boundaryFields<Vector>& bField, const Field<Vector>& internalField)
{
    using executor = typename CPUExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refGrad = bField.refGrad().field();
    const BoundaryMesh& boundaryMesh = mesh_.boundaryMesh();
    auto faceCells = mesh_.boundaryMesh().faceCells(patchID_);
    const auto iField = internalField.field();

    for (int i = start_; i < end_; ++i)
    {
        s_value[i] = iField[faceCells[i]];
        s_refGrad[i] = Vector(0, 0, 0);
    }
}

}
