// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred::surfaceBoundary
{
namespace detail
{
// Without this function the compiler warns that calling a __host__ function
// from a __device__ function is not allowed
template<ValueType T>
void setFixedValue(
    DomainField<T>& domainField,
    const UnstructuredMesh& mesh,
    std::pair<size_t, size_t> range,
    T fixedValue
)
{
    auto refValue = domainField.boundaryField().refValue().span();
    auto value = domainField.boundaryField().value().span();
    auto internalValues = domainField.internalField().span();
    auto nInternalFaces = mesh.nInternalFaces();

    NeoFOAM::parallelFor(
        domainField.exec(),
        range,
        KOKKOS_LAMBDA(const size_t i) {
            refValue[i] = fixedValue;
            value[i] = fixedValue;
            internalValues[nInternalFaces + i] = fixedValue;
        }
    );
}
}

template<ValueType T>
class FixedValue : public SurfaceBoundaryFactory<T>::template Register<FixedValue<T>>
{
    using Base = SurfaceBoundaryFactory<T>::template Register<FixedValue<T>>;

public:

    FixedValue(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID), mesh_(mesh), fixedValue_(dict.get<T>("fixedValue"))
    {}

    virtual void correctBoundaryCondition(DomainField<T>& domainField) override
    {
        detail::setFixedValue(domainField, mesh_, this->range(), fixedValue_);
    }

    static std::string name() { return "fixedValue"; }

    static std::string doc() { return "Set a fixed value on the boundary"; }

    static std::string schema() { return "none"; }

private:

    const UnstructuredMesh& mesh_;
    T fixedValue_;
};

}
