// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

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
void setGradientValue(
    DomainField<T>& domainField,
    const UnstructuredMesh& mesh,
    std::pair<size_t, size_t> range,
    size_t patchID,
    T fixedGradient
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
            // operator / is not defined for all Ts
            value[i] =
                iField[static_cast<size_t>(faceCells[i])] + fixedGradient * (1 / deltaCoeffs[i]);
        }
    );
}
}

template<ValueType T>
class FixedGradient : public VolumeBoundaryFactory<T>::template Register<FixedGradient<T>>
{
    using Base = VolumeBoundaryFactory<T>::template Register<FixedGradient<T>>;

public:

    using FixedGradientType = FixedGradient<T>;

    FixedGradient(const UnstructuredMesh& mesh, const Dictionary& dict, std::size_t patchID)
        : Base(mesh, dict, patchID), mesh_(mesh), fixedGradient_(dict.get<T>("fixedGradient"))
    {}

    virtual void correctBoundaryCondition(DomainField<T>& domainField) override
    {
        detail::setGradientValue(
            domainField, mesh_, this->range(), this->patchID(), fixedGradient_
        );
    }

    static std::string name() { return "fixedGradient"; }

    static std::string doc() { return "Set a fixed gradient on the boundary."; }

    static std::string schema() { return "none"; }


private:

    const UnstructuredMesh& mesh_;
    T fixedGradient_;
};

}
