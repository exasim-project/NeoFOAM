// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "primitives/scalar.hpp"
#include "primitives/label.hpp"
#include "deviceField.hpp"
#include "boundaryFields.hpp"

#include "NeoFOAM/blas/executor/executor.hpp"
#include "NeoFOAM/blas/span.hpp"

namespace NeoFOAM
{
    
    template <typename T>
    class domainField
    {
    /**
     * @class domainField
     * @brief Represents the domain fields for a computational domain.
     *
     * The domainField class stores the internal fields and boundary information for a computational domain.
     * It provides access to the computed values, reference values, value fractions, reference gradients, boundary types,
     * offsets, and the number of boundaries and boundary faces.
     */
        public:

            domainField(int nCells, int nBoundaryFaces, int nBoundaries, const executor& exec)
                :  exec_(exec),
                   internalField_(nCells,exec),
                   boundaryFields_(nBoundaryFaces, nBoundaries,exec)
            {
            }


            
            domainField(const domainField<T> &rhs)
                :  exec_(rhs.exec_),
                   internalField_(rhs.internalField_),
                   boundaryFields_(rhs.boundaryFields_)
            {
            }

            
            domainField(domainField<T> &&rhs)
                :  exec_(std::move(rhs.exec_)),
                   internalField_(std::move(rhs.internalField_)),
                   boundaryFields_(std::move(rhs.boundaryFields_))
            {
            }

            
            domainField<T> &operator=(const domainField<T> &rhs)
            {
                internalField_ = rhs.internalField_;
                boundaryFields_ = rhs.boundaryFields_;
                return *this;
            }

            
            domainField<T> &operator=(domainField<T> &&rhs)
            {
                internalField_ = std::move(rhs.internalField_);
                boundaryFields_ = std::move(rhs.boundaryFields_);
                return *this;
            }

            
            const Field< T>& internalField() const
            {
                return internalField_;
            }

            
            Field< T>& internalField()
            {
                return internalField_;
            }

            
            const boundaryFields< T>& boundaryField() const
            {
                return boundaryFields_;
            }

            
            boundaryFields< T>& boundaryField()
            {
                return boundaryFields_;
            }

        private:
            executor exec_;
            Field<T > internalField_; 
            boundaryFields<T > boundaryFields_;

    };

         
} // namespace NeoFOAM
