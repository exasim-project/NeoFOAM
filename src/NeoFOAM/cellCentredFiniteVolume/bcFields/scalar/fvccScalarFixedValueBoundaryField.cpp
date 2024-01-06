#include "NeoFOAM/cellCentredFiniteVolume/bcFields/scalar/fvccScalarFixedValueBoundaryField.hpp"

namespace NeoFOAM
{
    fvccScalarFixedValueBoundaryField::fvccScalarFixedValueBoundaryField(int start, int end, scalar uniformValue)
        :  fvccBoundaryField<scalar>(start, end),
           uniformValue_(uniformValue)
    {
    }

    void fvccScalarFixedValueBoundaryField::correctBoundaryConditions(boundaryFields<scalar> &field)
    {
        scalarField& value = field.value();
        scalarField& refValue = field.refValue();
        scalar uniformValue = uniformValue_;
        Kokkos::parallel_for("fvccScalarFixedValueBoundaryField", Kokkos::RangePolicy<>(start_, end_), KOKKOS_LAMBDA (const int i)
        {
            value(i) = uniformValue;
            refValue(i) = uniformValue;
        });
    }
}