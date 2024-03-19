#include "NeoFOAM/cellCentredFiniteVolume/bcFields/scalar/fvccScalarZeroGradientBoundaryField.hpp"

namespace NeoFOAM
{
fvccScalarZeroGradientBoundaryField::fvccScalarZeroGradientBoundaryField(int start, int end, scalar uniformValue)
    : fvccBoundaryField<scalar>(start, end)
{
}

void fvccScalarZeroGradientBoundaryField::correctBoundaryConditions(boundaryFields<scalar>& field)
{
    ZeroGradientBCKernel kernel_(start_, end_);
    std::visit([&](const auto& exec)
               { kernel_(exec, field); },
               field.exec());
}

void ZeroGradientBCKernel::operator()(const GPUExecutor& exec, boundaryFields<scalar>& bField, const unstructuredMesh& mesh)
{
    using executor = typename GPUExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refGrad = bField.refGrad().field();
    Kokkos::parallel_for(
        "fvccScalarZeroGradientBoundaryField", Kokkos::RangePolicy<executor>(start_, end_), KOKKOS_LAMBDA(const int i) {
            s_value[i] = 0;
            s_refGrad[i] = 0;
        }
    );
}

void ZeroGradientBCKernel::operator()(const OMPExecutor& exec, boundaryFields<scalar>& bField, const unstructuredMesh& mesh)
{
    using executor = typename OMPExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refGrad = bField.refGrad().field();
    Kokkos::parallel_for(
        "fvccScalarZeroGradientBoundaryField", Kokkos::RangePolicy<executor>(start_, end_), KOKKOS_LAMBDA(const int i) {
            s_value[i] = 0;
            s_refGrad[i] = 0;
        }
    );
}

void ZeroGradientBCKernel::operator()(const CPUExecutor& exec, boundaryFields<scalar>& bField, const unstructuredMesh& mesh)
{
    using executor = typename CPUExecutor::exec;
    auto s_value = bField.value().field();
    auto s_refGrad = bField.refGrad().field();

    for (int i = start_; i < end_; ++i)
    {
        s_value[i] = 0;
        s_refGrad[i] = 0;
    }
}

}