// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once
#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryBase.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{
template<typename ValueType>
class Calculated : public VolumeBoundaryFactory<ValueType>
{

public:

    using CalculatedType = Calculated<ValueType>;

    Calculated(std::size_t start, std::size_t end, std::size_t patchID)
        : VolumeBoundaryFactory<ValueType>(), start_(start), end_(end), patchID_(patchID)
    {
        VolumeBoundaryFactory<ValueType>::template registerClass<CalculatedType>();
    }

    static std::unique_ptr<VolumeBoundaryFactory<ValueType>>
    create(const NeoFOAM::UnstructuredMesh& mesh, const NeoFOAM::Dictionary& dict, int patchID)
    {
        std::size_t start = dict.get<std::size_t>("start");
        std::size_t end = dict.get<std::size_t>("end");
        return std::make_unique<CalculatedType>(start, end, patchID);
    }

    virtual void correctBoundaryCondition(NeoFOAM::DomainField<ValueType>& domainField) override {}

    static std::string name() { return "calculated"; }

private:

    std::size_t start_;
    std::size_t end_;
    std::size_t patchID_;
};

}
