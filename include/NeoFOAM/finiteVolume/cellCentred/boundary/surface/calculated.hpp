// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred::surfaceBoundary
{

template<typename ValueType>
class Calculated :
    public SurfaceBoundaryFactory<ValueType>::template Register<Calculated<ValueType>>
{
    using Base = SurfaceBoundaryFactory<ValueType>::template Register<Calculated<ValueType>>;

public:

    using CalculatedType = Calculated<ValueType>;

    Calculated(
        const UnstructuredMesh& mesh, [[maybe_unused]] const Dictionary& dict, size_t patchID
    )
        : Base(mesh, dict, patchID)
    {}

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) override {}

    static std::string name() { return "calculated"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }
};
}
