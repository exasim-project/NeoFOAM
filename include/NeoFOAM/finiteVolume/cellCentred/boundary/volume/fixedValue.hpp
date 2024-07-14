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

namespace impl
{
// Without this function the compiler warns that calling a __host__ function
// from
template<typename ValueType>
void setFixedValue(
    DomainField<ValueType>& domainField, std::pair<size_t, size_t> range, ValueType fixedValue
)
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

template<typename ValueType>
class FixedValue : public VolumeBoundaryFactory<ValueType>::template Register<FixedValue<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<FixedValue<ValueType>>;

public:

    FixedValue(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID), fixedValue_(dict.get<ValueType>("fixedValue"))
    {}

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) final
    {
        impl::setFixedValue(domainField, this->range(), fixedValue_);
    }

    static std::string name() { return "fixedValue"; }

    static std::string doc() { return "Set a fixed value on the boundary"; }

    static std::string schema() { return "none"; }

private:

    ValueType fixedValue_;
};

}
