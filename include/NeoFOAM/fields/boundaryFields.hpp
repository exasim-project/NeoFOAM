// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>

#include <iostream>

#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/fields/field.hpp"

namespace NeoFOAM
{


/**
 * @class BoundaryFields
 * @brief Represents the boundary fields for a computational domain.
 *
 * The BoundaryFields class stores the boundary conditions and related
 * information for a computational domain. It provides access to the computed
 * values, reference values, value fractions, reference gradients, boundary
 * types, offsets, and the number of boundaries and boundary faces.
 *
 * @tparam ValueType The type of the underlying field values
 */
template<typename T>
class BoundaryFields
{

public:

    /**
     * @brief Copy constructor.
     * @param rhs The boundaryFields object to be copied.
     */
    BoundaryFields(const BoundaryFields<T>& rhs)
        : exec_(rhs.exec_), value_(rhs.value_), refValue_(rhs.refValue_),
          valueFraction_(rhs.valueFraction_), refGrad_(rhs.refGrad_),
          boundaryTypes_(rhs.boundaryTypes_), offset_(rhs.offset_), nBoundaries_(rhs.nBoundaries_),
          nBoundaryFaces_(rhs.nBoundaryFaces_)
    {}


    BoundaryFields(const Executor& exec, int nBoundaryFaces, int nBoundaries)
        : exec_(exec), value_(exec, nBoundaryFaces), refValue_(exec, nBoundaryFaces),
          valueFraction_(exec, nBoundaryFaces), refGrad_(exec, nBoundaryFaces),
          boundaryTypes_(exec, nBoundaries), offset_(exec, nBoundaries + 1),
          nBoundaries_(nBoundaries), nBoundaryFaces_(nBoundaryFaces)
    {}


    /** @copydoc BoundaryFields::value()*/
    const NeoFOAM::Field<T>& value() const { return value_; }

    /**
     * @brief Get the view storing the computed values from the boundary
     * condition.
     * @return The view storing the computed values.
     */
    NeoFOAM::Field<T>& value() { return value_; }

    /** @copydoc BoundaryFields::refValue()*/
    const NeoFOAM::Field<T>& refValue() const { return refValue_; }

    /**
     * @brief Get the view storing the Dirichlet boundary values.
     * @return The view storing the Dirichlet boundary values.
     */
    NeoFOAM::Field<T>& refValue() { return refValue_; }

    /** @copydoc BoundaryFields::valueFraction()*/
    const NeoFOAM::Field<scalar>& valueFraction() const { return valueFraction_; }

    /**
     * @brief Get the view storing the fraction of the boundary value.
     * @return The view storing the fraction of the boundary value.
     */
    NeoFOAM::Field<scalar>& valueFraction() { return refValue_; }

    /** @copydoc BoundaryFields::refGrad()*/
    const NeoFOAM::Field<T>& refGrad() const { return refGrad_; }

    /**
     * @brief Get the view storing the Neumann boundary values.
     * @return The view storing the Neumann boundary values.
     */
    NeoFOAM::Field<T>& refGrad() { return refGrad_; }

    /**
     * @brief Get the view storing the boundary types.
     * @return The view storing the boundary types.
     */
    const NeoFOAM::Field<int>& boundaryTypes() const { return boundaryTypes_; }

    /**
     * @brief Get the view storing the offsets of each boundary.
     * @return The view storing the offsets of each boundary.
     */
    const NeoFOAM::Field<localIdx>& offset() const { return offset_; }

    /**
     * @brief Get the number of boundaries.
     * @return The number of boundaries.
     */
    int nBoundaries() const { return nBoundaries_; }

    /**
     * @brief Get the number of boundary faces.
     * @return The number of boundary faces.
     */
    int nBoundaryFaces() const { return nBoundaryFaces_; }

    const Executor& exec() { return exec_; }

private:

    Executor exec_;                        ///< The executor on which the field is stored
    NeoFOAM::Field<T> value_;              ///< The Field storing the computed values from the
                                           ///< boundary condition.
    NeoFOAM::Field<T> refValue_;           ///< The Field storing the Dirichlet boundary values.
    NeoFOAM::Field<scalar> valueFraction_; ///< The Field storing the fraction of
                                           ///< the boundary value.
    NeoFOAM::Field<T> refGrad_;            ///< The Field storing the Neumann boundary values.
    NeoFOAM::Field<int> boundaryTypes_;    ///< The Field storing the boundary types.
    NeoFOAM::Field<localIdx> offset_;      ///< The Field storing the offsets of each boundary.
    int nBoundaries_;                      ///< The number of boundaries.
    int nBoundaryFaces_;                   ///< The number of boundary faces.
};

}
