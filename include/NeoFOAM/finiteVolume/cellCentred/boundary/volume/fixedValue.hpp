// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{
// Without this function the compiler warns that calling a __host__ function
// from
template<ValueType T>
void setFixedValue(DomainField<T>& domainField, std::pair<size_t, size_t> range, T fixedValue)
{
    auto refValue = domainField.boundaryField().refValue().span();
    auto value = domainField.boundaryField().value().span();

    NeoFOAM::parallelFor(
        domainField.exec(),
        range,
        KOKKOS_LAMBDA(const size_t i) {
            refValue[i] = fixedValue;
            value[i] = fixedValue;
        }
    );
}

}

template<ValueType T>
class FixedValue : public VolumeBoundaryFactory<T>::template Register<FixedValue<T>>
{
    using Base = VolumeBoundaryFactory<T>::template Register<FixedValue<T>>;

public:

    FixedValue(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID), fixedValue_(dict.get<T>("fixedValue"))
    {}

    virtual void correctBoundaryCondition(DomainField<T>& domainField) final
    {
        detail::setFixedValue(domainField, this->range(), fixedValue_);
    }

    static std::string name() { return "fixedValue"; }

    static std::string doc() { return "Set a fixed value on the boundary"; }

    static std::string schema() { return "none"; }

private:

    T fixedValue_;
};

}
