// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoFOAM::finiteVolume::cellCentred::volumeBoundary
{

// TODO move to source file
namespace detail
{
// Without this function the compiler warns that calling a __host__ function
// from a __device__ function is not allowed
// NOTE: patchID was removed since it was unused
// I guess it was replaced by range
template<typename ValueType>
void extrapolateValue(
    DomainField<ValueType>& domainField,
    const UnstructuredMesh& mesh,
    std::pair<size_t, size_t> range
)
{
    const auto iField = domainField.internalField().view();

    auto [refGradient, value, valueFraction, refValue, faceCells] = spans(
        domainField.boundaryField().refGrad(),
        domainField.boundaryField().value(),
        domainField.boundaryField().valueFraction(),
        domainField.boundaryField().refValue(),
        mesh.boundaryMesh().faceCells()
    );


    NeoFOAM::parallelFor(
        domainField.exec(),
        range,
        KOKKOS_LAMBDA(const size_t i) {
            // operator / is not defined for all ValueTypes
            ValueType internalCellValue = iField[static_cast<size_t>(faceCells[i])];
            value[i] = internalCellValue;
            valueFraction[i] = 1.0;          // only use refValue
            refValue[i] = internalCellValue; // not used
            refGradient[i] = zero<ValueType>();
        },
        "extrapolateValue"
    );
}
}

template<typename ValueType>
class Extrapolated :
    public VolumeBoundaryFactory<ValueType>::template Register<Extrapolated<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<Extrapolated<ValueType>>;

public:

    using ExtrapolatedType = Extrapolated<ValueType>;

    Extrapolated(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition([[maybe_unused]] DomainField<ValueType>& domainField
    ) final
    {
        detail::extrapolateValue(domainField, mesh_, this->range());
    }

    static std::string name() { return "extrapolated"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Extrapolated>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};
}
