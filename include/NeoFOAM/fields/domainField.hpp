// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>

#include <iostream>

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/types.hpp"
#include "NeoFOAM/fields/boundaryFields.hpp"

#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoFOAM
{

/**
 * @class DomainField
 * @brief Represents the domain fields for a computational domain.
 *
 * The DomainField class stores the internal fields and boundary information for
 * a computational domain. It provides access to the computed values, reference
 * values, value fractions, reference gradients, boundary types, offsets, and
 * the number of boundaries and boundary faces.
 *
 * @tparam ValueType The type of the underlying field values
 */
template<ValueType T>
class DomainField
{
public:

    DomainField(const Executor& exec)
        : exec_(exec), internalField_(exec, 0), boundaryFields_(exec, 0, 0)
    {}

    DomainField(const Executor& exec, size_t nCells, size_t nBoundaryFaces, size_t nBoundaries)
        : exec_(exec), internalField_(exec, nCells),
          boundaryFields_(exec, nBoundaryFaces, nBoundaries)
    {}


    DomainField(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), internalField_(exec, mesh.nCells()),
          boundaryFields_(exec, mesh.nBoundaryFaces(), mesh.nBoundaries())
    {}


    DomainField(const ::NeoFOAM::DomainField<T>& rhs)
        : exec_(rhs.exec_), internalField_(rhs.internalField_), boundaryFields_(rhs.boundaryFields_)
    {}


    DomainField(DomainField<T>&& rhs)
        : exec_(std::move(rhs.exec_)), internalField_(std::move(rhs.internalField_)),
          boundaryFields_(std::move(rhs.boundaryFields_))
    {}


    DomainField<T>& operator=(const ::NeoFOAM::DomainField<T>& rhs)
    {
        internalField_ = rhs.internalField_;
        boundaryFields_ = rhs.boundaryFields_;
        return *this;
    }


    DomainField<T>& operator=(DomainField<T>&& rhs)
    {
        internalField_ = std::move(rhs.internalField_);
        boundaryFields_ = std::move(rhs.boundaryFields_);
        return *this;
    }


    const Field<T>& internalField() const { return internalField_; }


    Field<T>& internalField() { return internalField_; }


    const BoundaryFields<T>& boundaryField() const { return boundaryFields_; }


    BoundaryFields<T>& boundaryField() { return boundaryFields_; }

    const Executor& exec() const { return exec_; }

private:

    Executor exec_; ///< The executor on which the field is stored
    Field<T> internalField_;
    BoundaryFields<T> boundaryFields_;
};


} // namespace NeoFOAM
