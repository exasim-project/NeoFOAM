// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{
// Without this function the compiler warns that calling a __host__ function
// from a __device__ function is not allowed
template<typename ValueType>
void setGradientValue(
    DomainField<ValueType>& domainField,
    const UnstructuredMesh& mesh,
    std::pair<size_t, size_t> range,
    size_t patchID,
    ValueType fixedGradient
)
{
    const auto iField = domainField.internalField().span();
    auto refGradient = domainField.boundaryField().refGrad().span();
    auto value = domainField.boundaryField().value().span();
    auto faceCells = mesh.boundaryMesh().faceCells(static_cast<localIdx>(patchID));
    auto deltaCoeffs = mesh.boundaryMesh().deltaCoeffs(static_cast<localIdx>(patchID));

    NeoFOAM::parallelFor(
        domainField.exec(),
        range,
        KOKKOS_LAMBDA(const size_t i) {
            refGradient[i] = fixedGradient;
            // operator / is not defined for all ValueTypes
            value[i] =
                iField[static_cast<size_t>(faceCells[i])] + fixedGradient * (1 / deltaCoeffs[i]);
        }
    );
}
}

template<typename ValueType>
class FixedGradient :
    public VolumeBoundaryFactory<ValueType>::template Register<FixedGradient<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<FixedGradient<ValueType>>;

public:

    using FixedGradientType = FixedGradient<ValueType>;

    FixedGradient(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID), mesh_(mesh),
          fixedGradient_(dict.get<ValueType>("fixedGradient"))
    {}

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) final
    {
        detail::setGradientValue(
            domainField, mesh_, this->range(), this->patchID(), fixedGradient_
        );
    }

    static std::string name() { return "fixedGradient"; }

    static std::string doc() { return "Set a fixed gradient on the boundary."; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<FixedGradient>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    ValueType fixedGradient_;
};

}
