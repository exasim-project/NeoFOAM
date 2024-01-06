// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/blas/field.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "NeoFOAM/blas/boundaryFields.hpp"

namespace NeoFOAM
{

    template <typename T>
    class fvccBoundaryField
    {

        public:

            fvccBoundaryField(int start, int end)
                :  start_(start),
                   end_(end),
                   size_(end - start)

            {
                
            };

            virtual void correctBoundaryConditions(boundaryFields<T> &field)
            {
                
            };

        protected:
            
                int start_;
                int end_;
                int size_;

    };
};