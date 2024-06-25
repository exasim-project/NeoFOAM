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
class Empty : public VolumeBoundaryFactory<ValueType>, public BoundaryPatchMixin
{

public:

    using EmptyType = Empty<ValueType>;

    Empty(const UnstructuredMesh& mesh, std::size_t patchID)
        : VolumeBoundaryFactory<ValueType>(), BoundaryPatchMixin(mesh, patchID)
    {
        VolumeBoundaryFactory<ValueType>::template registerClass<EmptyType>();
    }

    static std::unique_ptr<VolumeBoundaryFactory<ValueType>>
    create(const UnstructuredMesh& mesh, const Dictionary&, std::size_t patchID)
    {
        return std::make_unique<EmptyType>(mesh, patchID);
    }

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) override {}

    static std::string name() { return "empty"; }
};

}
