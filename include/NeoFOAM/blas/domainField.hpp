// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "primitives/scalar.hpp"
#include "primitives/label.hpp"
#include "deviceField.hpp"
#include "boundaryFields.hpp"

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

            domainField(int nCells, int nBoundaryFaces, int nBoundaries)
                :  internalField_("internalField", nCells),
                   boundaryFields_(nBoundaryFaces, nBoundaries)
            {
            }


            KOKKOS_FUNCTION
            domainField(const domainField<T> &rhs)
                :  internalField_(rhs.internalField_),
                   boundaryFields_(rhs.boundaryFields_)
            {
            }

            KOKKOS_FUNCTION
            domainField(domainField<T> &&rhs)
                :  internalField_(std::move(rhs.internalField_)),
                   boundaryFields_(std::move(rhs.boundaryFields_))
            {
            }

            KOKKOS_FUNCTION
            domainField<T> &operator=(const domainField<T> &rhs)
            {
                internalField_ = rhs.internalField_;
                boundaryFields_ = rhs.boundaryFields_;
                return *this;
            }

            KOKKOS_FUNCTION
            domainField<T> &operator=(domainField<T> &&rhs)
            {
                internalField_ = std::move(rhs.internalField_);
                boundaryFields_ = std::move(rhs.boundaryFields_);
                return *this;
            }

            KOKKOS_FUNCTION
            const deviceField< T>& internalField() const
            {
                return internalField_;
            }

            KOKKOS_FUNCTION
            deviceField< T>& internalField()
            {
                return internalField_;
            }

            KOKKOS_FUNCTION
            const boundaryFields< T>& boundaryField() const
            {
                return boundaryFields_;
            }

            KOKKOS_FUNCTION
            boundaryFields< T>& boundaryField()
            {
                return boundaryFields_;
            }

        private:

            deviceField<T > internalField_; 
            boundaryFields<T > boundaryFields_;

    };

         
} // namespace NeoFOAM
