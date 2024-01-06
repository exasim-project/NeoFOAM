// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/blas/primitives/scalar.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "Kokkos_Core.hpp"

namespace NeoFOAM
{
    class fvccScalarFixedValueBoundaryField : public fvccBoundaryField<scalar>
    {
        public:
            fvccScalarFixedValueBoundaryField(int start, int end, scalar uniformValue);

            void correctBoundaryConditions(boundaryFields<scalar> &field);

        private:
            scalar uniformValue_;
    };
};