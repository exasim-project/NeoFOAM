// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorFixedValueBoundaryField.hpp"

namespace NeoFOAM
{
fvccVectorFixedValueBoundaryField::fvccVectorFixedValueBoundaryField(
    const UnstructuredMesh& mesh, int patchID, Vector uniformValue
)
    : fvccBoundaryField<Vector>(mesh, patchID), uniformValue_(uniformValue)
{}

void fvccVectorFixedValueBoundaryField::correctBoundaryConditions(
    BoundaryFields<Vector>& bfield, const Field<Vector>& internalField
)
{
    fixedVectorValueBCKernel kernel_(mesh_, patchID_, start_, end_, uniformValue_);
    std::visit([&](const auto& exec) { kernel_(exec, bfield, internalField); }, bfield.exec());
}

void fixedVectorValueBCKernel::operator()(
    const GPUExecutor& exec, BoundaryFields<Vector>& bField, const Field<Vector>& internalField
)
{
    using executor = typename GPUExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refValue = bField.refValue().field();
    Vector uniformValue = uniformValue_;
    Kokkos::parallel_for(
        "fvccVectorFixedValueBoundaryField",
        Kokkos::RangePolicy<executor>(start_, end_),
        KOKKOS_LAMBDA(const int i) {
            s_value[i] = uniformValue;
            s_refValue[i] = uniformValue;
        }
    );
}

void fixedVectorValueBCKernel::operator()(
    const OMPExecutor& exec, BoundaryFields<Vector>& bField, const Field<Vector>& internalField
)
{
    using executor = typename OMPExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refValue = bField.refValue().field();
    Vector uniformValue = uniformValue_;
    Kokkos::parallel_for(
        "fvccVectorFixedValueBoundaryField",
        Kokkos::RangePolicy<executor>(start_, end_),
        KOKKOS_LAMBDA(const int i) {
            s_value[i] = uniformValue;
            s_refValue[i] = uniformValue;
        }
    );
}

void fixedVectorValueBCKernel::operator()(
    const CPUExecutor& exec, BoundaryFields<Vector>& bField, const Field<Vector>& internalField
)
{
    using executor = typename CPUExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refValue = bField.refValue().field();

    Vector uniformValue = uniformValue_;

    for (int i = start_; i < end_; ++i)
    {
        s_value[i] = uniformValue;
        s_refValue[i] = uniformValue;
    }
}

}
