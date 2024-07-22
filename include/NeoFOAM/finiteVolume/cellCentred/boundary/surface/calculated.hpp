// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred::surfaceBoundary
{

template<ValueType T>
class Calculated : public SurfaceBoundaryFactory<T>::template Register<Calculated<T>>
{
    using Base = SurfaceBoundaryFactory<T>::template Register<Calculated<T>>;

public:

    using CalculatedType = Calculated<T>;

    Calculated(
        const UnstructuredMesh& mesh, [[maybe_unused]] const Dictionary& dict, std::size_t patchID
    )
        : Base(mesh, dict, patchID)
    {}

    virtual void correctBoundaryCondition(DomainField<T>& domainField) override {}

    static std::string name() { return "calculated"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }
};
}
