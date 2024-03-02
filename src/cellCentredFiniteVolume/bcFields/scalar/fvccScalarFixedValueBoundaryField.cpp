#include "NeoFOAM/cellCentredFiniteVolume/bcFields/scalar/fvccScalarFixedValueBoundaryField.hpp"

namespace NeoFOAM
{
fvccScalarFixedValueBoundaryField::fvccScalarFixedValueBoundaryField(int start, int end, scalar uniformValue)
    : fvccBoundaryField<scalar>(start, end),
      uniformValue_(uniformValue)
{
}

void fvccScalarFixedValueBoundaryField::correctBoundaryConditions(boundaryFields<scalar>& field)
{
    fixedValueBCKernel kernel_(uniformValue_, start_, end_);
    std::visit([&](const auto& exec)
               { kernel_(exec, field); },
               field.exec());
}

void fixedValueBCKernel::operator()(const GPUExecutor& exec, boundaryFields<scalar>& bField)
{
    using executor = typename GPUExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refValue = bField.refValue().field();
    scalar uniformValue = uniformValue_;
    Kokkos::parallel_for(
        "fvccScalarFixedValueBoundaryField", Kokkos::RangePolicy<executor>(start_, end_), KOKKOS_LAMBDA(const int i) {
            s_value[i] = uniformValue;
            s_refValue[i] = uniformValue;
        }
    );
}

void fixedValueBCKernel::operator()(const OMPExecutor& exec, boundaryFields<scalar>& bField)
{
    using executor = typename OMPExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refValue = bField.refValue().field();
    scalar uniformValue = uniformValue_;
    Kokkos::parallel_for(
        "fvccScalarFixedValueBoundaryField", Kokkos::RangePolicy<executor>(start_, end_), KOKKOS_LAMBDA(const int i) {
            s_value[i] = uniformValue;
            s_refValue[i] = uniformValue;
        }
    );
}

void fixedValueBCKernel::operator()(const CPUExecutor& exec, boundaryFields<scalar>& bField)
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