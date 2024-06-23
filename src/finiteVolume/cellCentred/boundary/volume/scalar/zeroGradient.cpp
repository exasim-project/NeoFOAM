// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred.hpp"

namespace NeoFOAM
{
fvccScalarZeroGradientBoundaryField::fvccScalarZeroGradientBoundaryField(
    const UnstructuredMesh& mesh, int patchID
)
    : fvccBoundaryField<scalar>(mesh, patchID)
{}

void fvccScalarZeroGradientBoundaryField::correctBoundaryConditions(
    BoundaryFields<scalar>& bfield, const Field<scalar>& internalField
)
{
    ZeroGradientBCKernel kernel_(mesh_, patchID_, start_, end_);
    std::visit([&](const auto& exec) { kernel_(exec, bfield, internalField); }, bfield.exec());
}


void ZeroGradientBCKernel::operator()(
    const GPUExecutor& exec, BoundaryFields<scalar>& bField, const Field<scalar>& internalField
)
{
    using executor = typename GPUExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refGrad = bField.refGrad().field();
    auto faceCells = mesh_.boundaryMesh().faceCells(patchID_);
    const auto iField = internalField.field();
    Kokkos::parallel_for(
        "fvccScalarZeroGradientBoundaryField",
        Kokkos::RangePolicy<executor>(start_, end_),
        KOKKOS_LAMBDA(const int i) {
            s_value[i] = iField[faceCells[i]];
            s_refGrad[i] = 0;
        }
    );
}

void ZeroGradientBCKernel::operator()(
    const OMPExecutor& exec, BoundaryFields<scalar>& bField, const Field<scalar>& internalField
)
{
    using executor = typename OMPExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refGrad = bField.refGrad().field();
    auto faceCells = mesh_.boundaryMesh().faceCells(patchID_);
    const auto iField = internalField.field();
    Kokkos::parallel_for(
        "fvccScalarZeroGradientBoundaryField",
        Kokkos::RangePolicy<executor>(start_, end_),
        KOKKOS_LAMBDA(const int i) {
            s_value[i] = iField[faceCells[i]];
            s_refGrad[i] = 0;
        }
    );
}

void ZeroGradientBCKernel::operator()(
    const CPUExecutor& exec, BoundaryFields<scalar>& bField, const Field<scalar>& internalField
)
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
        s_refGrad[i] = 0;
    }
}

}