// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "primitives/scalar.hpp"
#include "primitives/label.hpp"
#include "deviceField.hpp"

namespace NeoFOAM
{
    // enum class boundaryType used in a kokkos view to categorize boundary
    // types


    template <typename T>
    class boundaryFields
    {
    /**
     * @class boundaryFields
     * @brief Represents the boundary fields for a computational domain.
     *
     * The boundaryFields class stores the boundary conditions and related information for a computational domain.
     * It provides access to the computed values, reference values, value fractions, reference gradients, boundary types,
     * offsets, and the number of boundaries and boundary faces.
     */
    
    public:

        KOKKOS_FUNCTION
        enum class boundaryTypes: int
        {
            Dirichlet,
            Neumann,
            Robin
        };
        /**
         * @brief Copy constructor.
         * @param rhs The boundaryFields object to be copied.
         */
        KOKKOS_FUNCTION
        boundaryFields(const boundaryFields<T> &rhs)
            :  value_(rhs.value_),
               refValue_(rhs.refValue_),
               valueFraction_(rhs.valueFraction_),
               refGrad_(rhs.refGrad_),
               boundaryTypes_(rhs.boundaryTypes_),
               offset_(rhs.offset_),
               nBoundaries_(rhs.nBoundaries_),
               nBoundaryFaces_(rhs.nBoundaryFaces_)
        {
        }

        
        boundaryFields(int nBoundaryFaces,int nBoundaries)
            :  value_("value", nBoundaryFaces),
               refValue_("refValue", nBoundaryFaces),
               valueFraction_("valueFraction", nBoundaryFaces),
               refGrad_("refGrad", nBoundaryFaces),
               boundaryTypes_("boundaryType", nBoundaries),
               offset_("offset", nBoundaries + 1),
               nBoundaries_(nBoundaries),
               nBoundaryFaces_(nBoundaryFaces)
        {
        }

        /**
         * @brief Get the view storing the computed values from the boundary condition.
         * @return The view storing the computed values.
         */
        const NeoFOAM::deviceField<T>&  value() const
        {
            return value_;
        }

        NeoFOAM::deviceField<T>&  value()
        {
            return value_;
        }

        /**
         * @brief Get the view storing the Dirichlet boundary values.
         * @return The view storing the Dirichlet boundary values.
         */
        const NeoFOAM::deviceField<T>&  refValue() const
        {
            return refValue_;
        }

        NeoFOAM::deviceField<T>&  refValue()
        {
            return refValue_;
        }

        /**
         * @brief Get the view storing the fraction of the boundary value.
         * @return The view storing the fraction of the boundary value.
         */
        const NeoFOAM::deviceField<scalar>& valueFraction() const
        {
            return valueFraction_;
        }

        NeoFOAM::deviceField<scalar>&  valueFraction()
        {
            return refValue_;
        }

        /**
         * @brief Get the view storing the Neumann boundary values.
         * @return The view storing the Neumann boundary values.
         */
        const NeoFOAM::deviceField<T>& refGrad() const
        {
            return refGrad_;
        }

        NeoFOAM::deviceField<T>& refGrad()
        {
            return refGrad_;
        }

        /**
         * @brief Get the view storing the boundary types.
         * @return The view storing the boundary types.
         */
        const NeoFOAM::deviceField<int >&  boundaryTypes() const
        {
            return boundaryTypes_;
        }

        /**
         * @brief Get the view storing the offsets of each boundary.
         * @return The view storing the offsets of each boundary.
         */
        const NeoFOAM::deviceField<localIdx>& offset() const
        {
            return offset_;
        }

        /**
         * @brief Get the number of boundaries.
         * @return The number of boundaries.
         */
        int nBoundaries() 
        {
            return nBoundaries_;
        }

        /**
         * @brief Get the number of boundary faces.
         * @return The number of boundary faces.
         */
        int nBoundaryFaces()
        {
            return nBoundaryFaces_;
        }

    private:
        NeoFOAM::deviceField<T > value_;                   ///< The view storing the computed values from the boundary condition.
        NeoFOAM::deviceField<T> refValue_;                ///< The view storing the Dirichlet boundary values.
        NeoFOAM::deviceField<scalar> valueFraction_;      ///< The view storing the fraction of the boundary value.
        NeoFOAM::deviceField<T> refGrad_;                 ///< The view storing the Neumann boundary values.
        NeoFOAM::deviceField<int > boundaryTypes_; ///< The view storing the boundary types.
        NeoFOAM::deviceField<localIdx> offset_;           ///< The view storing the offsets of each boundary.
        int nBoundaries_;                           ///< The number of boundaries.
        int nBoundaryFaces_;                        ///< The number of boundary faces.
    };
         
} // namespace NeoFOAM
