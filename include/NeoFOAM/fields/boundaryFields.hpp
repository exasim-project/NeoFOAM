// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "NeoFOAM/primitives/label.hpp"
#include "NeoFOAM/primitives/scalar.hpp"
#include "Field.hpp"

#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{
// enum class boundaryType used in a kokkos view to categorize boundary
// types


template<typename T>
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

    //
    // enum class boundaryTypes: int
    // {
    //     Dirichlet,
    //     Neumann,
    //     Robin
    // };
    /**
     * @brief Copy constructor.
     * @param rhs The boundaryFields object to be copied.
     */

    boundaryFields(const boundaryFields<T>& rhs)
        : exec_(rhs.exec_),
          value_(rhs.value_),
          refValue_(rhs.refValue_),
          valueFraction_(rhs.valueFraction_),
          refGrad_(rhs.refGrad_),
          boundaryTypes_(rhs.boundaryTypes_),
          offset_(rhs.offset_),
          nBoundaries_(rhs.nBoundaries_),
          nBoundaryFaces_(rhs.nBoundaryFaces_)
    {
    }


    boundaryFields(const executor& exec, int nBoundaryFaces, int nBoundaries)
        : exec_(exec),
          value_(exec, nBoundaryFaces),
          refValue_(exec, nBoundaryFaces),
          valueFraction_(exec, nBoundaryFaces),
          refGrad_(exec, nBoundaryFaces),
          boundaryTypes_(exec, nBoundaries),
          offset_(exec, nBoundaries + 1),
          nBoundaries_(nBoundaries),
          nBoundaryFaces_(nBoundaryFaces)
    {
    }

    /**
     * @brief Get the view storing the computed values from the boundary condition.
     * @return The view storing the computed values.
     */
    const NeoFOAM::Field<T>& value() const
    {
        return value_;
    }

    NeoFOAM::Field<T>& value()
    {
        return value_;
    }

    /**
     * @brief Get the view storing the Dirichlet boundary values.
     * @return The view storing the Dirichlet boundary values.
     */
    const NeoFOAM::Field<T>& refValue() const
    {
        return refValue_;
    }

    NeoFOAM::Field<T>& refValue()
    {
        return refValue_;
    }

    /**
     * @brief Get the view storing the fraction of the boundary value.
     * @return The view storing the fraction of the boundary value.
     */
    const NeoFOAM::Field<scalar>& valueFraction() const
    {
        return valueFraction_;
    }

    NeoFOAM::Field<scalar>& valueFraction()
    {
        return refValue_;
    }

    /**
     * @brief Get the view storing the Neumann boundary values.
     * @return The view storing the Neumann boundary values.
     */
    const NeoFOAM::Field<T>& refGrad() const
    {
        return refGrad_;
    }

    NeoFOAM::Field<T>& refGrad()
    {
        return refGrad_;
    }

    /**
     * @brief Get the view storing the boundary types.
     * @return The view storing the boundary types.
     */
    const NeoFOAM::Field<int>& boundaryTypes() const
    {
        return boundaryTypes_;
    }

    /**
     * @brief Get the view storing the offsets of each boundary.
     * @return The view storing the offsets of each boundary.
     */
    const NeoFOAM::Field<localIdx>& offset() const
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

    const executor& exec()
    {
        return exec_;
    }

private:

    executor exec_;
    NeoFOAM::Field<T> value_;              ///< The view storing the computed values from the boundary condition.
    NeoFOAM::Field<T> refValue_;           ///< The view storing the Dirichlet boundary values.
    NeoFOAM::Field<scalar> valueFraction_; ///< The view storing the fraction of the boundary value.
    NeoFOAM::Field<T> refGrad_;            ///< The view storing the Neumann boundary values.
    NeoFOAM::Field<int> boundaryTypes_;    ///< The view storing the boundary types.
    NeoFOAM::Field<localIdx> offset_;      ///< The view storing the offsets of each boundary.
    int nBoundaries_;                      ///< The number of boundaries.
    int nBoundaryFaces_;                   ///< The number of boundary faces.
};

} // namespace NeoFOAM
