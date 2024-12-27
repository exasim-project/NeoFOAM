// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>

#include <iostream>

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/boundaryFields.hpp"

#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoFOAM
{

/**
 * @class DomainField
 * @brief Represents a field that expands over the full computational domain ie the internal field
 * including boundary data.
 *
 * The DomainField class stores the internal field and the field data on the boundary for
 * a computational domain. It provides access to the computed values, reference
 * values, value fractions, reference gradients, boundary types, offsets, and
 * the number of boundaries and boundary faces.
 *
 * @tparam ValueType The type of the underlying field values
 */
template<typename ValueType>
class DomainField
{
public:

    /**
     * @brief Constructor for an uninitialized DomainField with zero size
     *
     * @param exec The executor object.
     */
    DomainField(const Executor& exec)
        : exec_(exec), internalField_(exec, 0), boundaryFields_(exec, 0, 0)
    {}

    /**
     * @brief Constructor for an uninitialized but preallocated DomainField with given size
     *
     * @param exec The executor object.
     */
    DomainField(const Executor& exec, size_t nCells, size_t nBoundaryFaces, size_t nBoundaries)
        : exec_(exec), internalField_(exec, nCells),
          boundaryFields_(exec, nBoundaryFaces, nBoundaries)
    {}

    /**
     * @brief Convenience constructor for an uninitialized but preallocated DomainField with given
     * size
     *
     * @param exec The executor object.
     * @param mesh The mesh from which the field sizes can be determined.
     */
    DomainField(const Executor& exec, const UnstructuredMesh& mesh)
        : DomainField(exec, mesh.nCells(), mesh.nBoundaryFaces(), mesh.nBoundaries())
    {}

    /**
     * @brief Constructor for a DomainField with preexisting internal and boundary data
     *
     * @param exec The executor object.
     */
    DomainField(
        const Executor& exec,
        const Field<ValueType>& internalField,
        const BoundaryFields<ValueType>& boundaryFields
    )
        : exec_(exec), internalField_(exec, internalField), boundaryFields_(exec, boundaryFields)
    {}

    /**
     * @brief Constructor for a DomainField with preexisting internal and boundary data
     *
     * @param exec The executor object.
     */
    DomainField(
        const Executor& exec,
        const Field<ValueType>&& internalField,
        const BoundaryFields<ValueType>&& boundaryFields
    )
        : exec_(exec), internalField_(exec, std::move(internalField)),
          boundaryFields_(exec, std::move(boundaryFields))
    {}

    /**
     * @brief Copy constructor for the DomainField
     *
     * @param rhs The field to copy.
     */
    DomainField(const DomainField<ValueType>& rhs)
        : exec_(rhs.exec_), internalField_(rhs.internalField_), boundaryFields_(rhs.boundaryFields_)
    {}

    /**
     * @brief Copy constructor for the DomainField
     *
     * @param rhs The field to copy.
     */
    DomainField(DomainField<ValueType>&& rhs)
        : exec_(std::move(rhs.exec_)), internalField_(std::move(rhs.internalField_)),
          boundaryFields_(std::move(rhs.boundaryFields_))
    {}

    DomainField<ValueType>& operator=(const DomainField<ValueType>& rhs)
    {
        internalField_ = rhs.internalField_;
        boundaryFields_ = rhs.boundaryFields_;
        return *this;
    }

    DomainField<ValueType>& operator=(DomainField<ValueType>&& rhs)
    {
        internalField_ = std::move(rhs.internalField_);
        boundaryFields_ = std::move(rhs.boundaryFields_);
        return *this;
    }


    const Field<ValueType>& internalField() const { return internalField_; }


    Field<ValueType>& internalField() { return internalField_; }


    const BoundaryFields<ValueType>& boundaryField() const { return boundaryFields_; }


    BoundaryFields<ValueType>& boundaryField() { return boundaryFields_; }

    const Executor& exec() const { return exec_; }

private:

    Executor exec_; ///< The executor on which the field is stored
    Field<ValueType> internalField_;
    BoundaryFields<ValueType> boundaryFields_;
};


} // namespace NeoFOAM
