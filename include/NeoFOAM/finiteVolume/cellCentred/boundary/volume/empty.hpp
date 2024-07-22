// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred::volumeBoundary
{

template<ValueType T>
class Empty : public VolumeBoundaryFactory<T>::template Register<Empty<T>>
{
    using Base = VolumeBoundaryFactory<T>::template Register<Empty<T>>;

public:

    Empty(
        const UnstructuredMesh& mesh, [[maybe_unused]] const Dictionary& dict, std::size_t patchID
    )
        : Base(mesh, dict, patchID)
    {}

    virtual void correctBoundaryCondition([[maybe_unused]] DomainField<T>& domainField) override {}

    static std::string name() { return "empty"; }

    static std::string doc() { return "Do nothing on the boundary."; }

    static std::string schema() { return "none"; }
};

}
