// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "primitives/scalar.hpp"
#include "primitives/label.hpp"

namespace NeoFOAM
{
    // enum class boundaryType used in a kokkos view to categorize boundary
    // types
    enum class boundaryType
    {
        Dirichlet,
        Neumann,
        Robin
    };

    template <typename T>
    class boundaryFields
    /**
     * @class boundaryFields
     * @brief Represents the boundary fields for a computational domain.
     *
     * The boundaryFields class stores the boundary conditions and related information for a computational domain.
     * It provides access to the computed values, reference values, value fractions, reference gradients, boundary types,
     * offsets, and the number of boundaries and boundary faces.
     */
    class boundaryFields {
    public:
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
               boundaryType_(rhs.boundaryType_),
               offset_(rhs.offset_),
               nBoundaries_(rhs.nBoundaries_),
               nBoundaryFaces_(rhs.nBoundaryFaces_)
        {
        }

        /**
         * @brief Constructor that initializes the boundaryFields object with the given components.
         * @param value The view storing the computed values from the boundary condition.
         * @param refValue The view storing the Dirichlet boundary values.
         * @param valueFraction The view storing the fraction of the boundary value.
         * @param refGrad The view storing the Neumann boundary values.
         * @param boundaryType The view storing the boundary types.
         * @param offset The view storing the offsets of each boundary.
         * @param nBoundaries The number of boundaries.
         * @param nBoundaryFaces The number of boundary faces.
         */
        KOKKOS_FUNCTION
        boundaryFields(const Kokkos::View<T *> &value,
                      const Kokkos::View<T *> &refValue,
                      const Kokkos::View<scalar *> &valueFraction,
                      const Kokkos::View<T *> &refGrad,
                      const Kokkos::View<boundaryType *> &boundaryType,
                      const Kokkos::View<localIdx *> &offset,
                      const int nBoundaries,
                      const int nBoundaryFaces)
            : value_(value),
              refValue_(refValue),
              valueFraction_(valueFraction),
              refGrad_(refGrad),
              boundaryType_(boundaryType),
              offset_(offset),
              nBoundaries_(nBoundaries),
              nBoundaryFaces_(nBoundaryFaces)
        {
        }

        /**
         * @brief Get the view storing the computed values from the boundary condition.
         * @return The view storing the computed values.
         */
        Kokkos::View<T *> value()
        {
            return value_;
        }

        /**
         * @brief Get the view storing the Dirichlet boundary values.
         * @return The view storing the Dirichlet boundary values.
         */
        Kokkos::View<T *> refValue()
        {
            return refValue_;
        }

        /**
         * @brief Get the view storing the fraction of the boundary value.
         * @return The view storing the fraction of the boundary value.
         */
        Kokkos::View<scalar *> valueFraction()
        {
            return valueFraction_;
        }

        /**
         * @brief Get the view storing the Neumann boundary values.
         * @return The view storing the Neumann boundary values.
         */
        Kokkos::View<T *> refGrad()
        {
            return refGrad_;
        }

        /**
         * @brief Get the view storing the boundary types.
         * @return The view storing the boundary types.
         */
        Kokkos::View<boundaryType *> boundaryType()
        {
            return boundaryType_;
        }

        /**
         * @brief Get the view storing the offsets of each boundary.
         * @return The view storing the offsets of each boundary.
         */
        Kokkos::View<localIdx *> offset()
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
        Kokkos::View<T *> value_;                   ///< The view storing the computed values from the boundary condition.
        Kokkos::View<T *> refValue_;                ///< The view storing the Dirichlet boundary values.
        Kokkos::View<scalar *> valueFraction_;      ///< The view storing the fraction of the boundary value.
        Kokkos::View<T *> refGrad_;                 ///< The view storing the Neumann boundary values.
        Kokkos::View<boundaryType *> boundaryType_; ///< The view storing the boundary types.
        Kokkos::View<localIdx *> offset_;           ///< The view storing the offsets of each boundary.
        int nBoundaries_;                           ///< The number of boundaries.
        int nBoundaryFaces_;                        ///< The number of boundary faces.
    };
         
} // namespace NeoFOAM
