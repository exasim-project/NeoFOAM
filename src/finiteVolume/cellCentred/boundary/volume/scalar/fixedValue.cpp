// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarFixedValueBoundaryField.hpp"

namespace NeoFOAM
{
fvccScalarFixedValueBoundaryField::fvccScalarFixedValueBoundaryField(
    const UnstructuredMesh& mesh, int patchID, scalar uniformValue
)
    : fvccBoundaryField<scalar>(mesh, patchID), uniformValue_(uniformValue)
{}

void fvccScalarFixedValueBoundaryField::correctBoundaryConditions(
    BoundaryFields<scalar>& bfield, const Field<scalar>& internalField
)
{
    fixedValueBCKernel kernel_(mesh_, patchID_, start_, end_, uniformValue_);
    std::visit([&](const auto& exec) { kernel_(exec, bfield, internalField); }, bfield.exec());
}

void fixedValueBCKernel::operator()(
    const GPUExecutor& exec, BoundaryFields<scalar>& bField, const Field<scalar>& internalField
)
{
    using executor = typename GPUExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refValue = bField.refValue().field();
    scalar uniformValue = uniformValue_;
    Kokkos::parallel_for(
        "fvccScalarFixedValueBoundaryField",
        Kokkos::RangePolicy<executor>(start_, end_),
        KOKKOS_LAMBDA(const int i) {
            s_value[i] = uniformValue;
            s_refValue[i] = uniformValue;
        }
    );
}

void fixedValueBCKernel::operator()(
    const OMPExecutor& exec, BoundaryFields<scalar>& bField, const Field<scalar>& internalField
)
{
    using executor = typename OMPExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refValue = bField.refValue().field();
    scalar uniformValue = uniformValue_;
    Kokkos::parallel_for(
        "fvccScalarFixedValueBoundaryField",
        Kokkos::RangePolicy<executor>(start_, end_),
        KOKKOS_LAMBDA(const int i) {
            s_value[i] = uniformValue;
            s_refValue[i] = uniformValue;
        }
    );
}

void fixedValueBCKernel::operator()(
    const CPUExecutor& exec, BoundaryFields<scalar>& bField, const Field<scalar>& internalField
)
{
    using executor = typename CPUExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refValue = bField.refValue().field();

    scalar uniformValue = uniformValue_;

    for (int i = start_; i < end_; ++i)
    {
        s_value[i] = uniformValue;
        s_refValue[i] = uniformValue;
    }
}

}
