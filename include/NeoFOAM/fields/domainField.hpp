// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "NeoFOAM/primitives/label.hpp"
#include "NeoFOAM/primitives/scalar.hpp"
#include "boundaryFields.hpp"

#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{

template<typename T>
class DomainField
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

    DomainField(const Executor& exec, int nCells, int nBoundaryFaces, int nBoundaries)
        : exec_(exec),
          internalField_(exec, nCells),
          boundaryFields_(exec, nBoundaryFaces, nBoundaries)
    {
    }


    DomainField(const ::NeoFOAM::DomainField<T>& rhs)
        : exec_(rhs.exec_),
          internalField_(rhs.internalField_),
          boundaryFields_(rhs.boundaryFields_)
    {
    }


    DomainField(DomainField<T>&& rhs)
        : exec_(std::move(rhs.exec_)),
          internalField_(std::move(rhs.internalField_)),
          boundaryFields_(std::move(rhs.boundaryFields_))
    {
    }


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


    const Field<T>& internalField() const
    {
        return internalField_;
    }


    Field<T>& internalField()
    {
        return internalField_;
    }


    const BoundaryFields<T>& boundaryField() const
    {
        return boundaryFields_;
    }


    BoundaryFields<T>& boundaryField()
    {
        return boundaryFields_;
    }

private:

    Executor exec_;
    Field<T> internalField_;
    BoundaryFields<T> boundaryFields_;
};


} // namespace NeoFOAM
