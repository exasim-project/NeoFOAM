// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/core/runtimeSelectionFactory.hpp"
#include "NeoFOAM/core/types.hpp"
#include "NeoFOAM/fields/domainField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/boundaryPatchMixin.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


template<ValueType T>
class SurfaceBoundaryFactory :
    public NeoFOAM::RuntimeSelectionFactory<
        SurfaceBoundaryFactory<T>,
        Parameters<const UnstructuredMesh&, const Dictionary&, size_t>>,
    public BoundaryPatchMixin
{
public:

    static std::string name() { return "SurfaceBoundaryFactory"; }

    SurfaceBoundaryFactory(
        const UnstructuredMesh& mesh, [[maybe_unused]] const Dictionary&, size_t patchID
    )
        : BoundaryPatchMixin(mesh, patchID) {};

    virtual void correctBoundaryCondition(DomainField<T>& domainField) = 0;
};


/**
 * @brief Represents a volume boundary field for a cell-centered finite volume method.
 *
 * @tparam T The data type of the field.
 */
template<ValueType T>
class SurfaceBoundary : public BoundaryPatchMixin
{
public:

    SurfaceBoundary(const UnstructuredMesh& mesh, const Dictionary& dict, size_t patchID)
        : BoundaryPatchMixin(
            static_cast<label>(mesh.boundaryMesh().offset()[patchID]),
            static_cast<label>(mesh.boundaryMesh().offset()[patchID + 1]),
            patchID
        ),
          boundaryCorrectionStrategy_(
              SurfaceBoundaryFactory<T>::create(dict.get<std::string>("type"), mesh, dict, patchID)
          )
    {}

    virtual void correctBoundaryCondition(DomainField<T>& domainField)
    {
        boundaryCorrectionStrategy_->correctBoundaryCondition(domainField);
    }

private:

    // NOTE needs full namespace to be not ambiguous
    std::unique_ptr<NeoFOAM::finiteVolume::cellCentred::SurfaceBoundaryFactory<T>>
        boundaryCorrectionStrategy_;
};

}
