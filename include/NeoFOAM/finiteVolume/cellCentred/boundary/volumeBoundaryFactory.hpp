// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/core/registerClass.hpp"
#include "NeoFOAM/fields.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/boundaryPatchMixin.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// forward declaration so we can use it to define the create function and the class manager
template<typename ValueType>
class VolumeBoundaryFactory;

// a detail namespace to prevent conflicts for surface boundaries
namespace cellCentred::VolumeBoundaryDetail
{

    // define the create function use to instantiate the derived classes
    template<typename ValueType>
    using CreateFunc = std::function<std::unique_ptr<VolumeBoundaryFactory<ValueType>>(
        const UnstructuredMesh&, const Dictionary, int
    )>;

    template<typename ValueType>
    using ClassRegistry =
        NeoFOAM::BaseClassRegistry<VolumeBoundaryFactory<ValueType>, CreateFunc<ValueType>>;
}

using namespace cellCentred::VolumeBoundaryDetail;

template<typename ValueType>
class VolumeBoundaryFactory : public ClassRegistry<ValueType>, public BoundaryPatchMixin
{
public:

    VolumeBoundaryFactory(const UnstructuredMesh& mesh, std::size_t patchID)
        : BoundaryPatchMixin(mesh, patchID) {};


    MAKE_CLASS_A_RUNTIME_FACTORY(VolumeBoundaryFactory<ValueType>, ClassRegistry<ValueType>, CreateFunc<ValueType>)

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) = 0;
};


/**
 * @brief Represents a surface boundary field for a cell-centered finite volume method.
 *
 * @tparam ValueType The data type of the field.
 */
template<typename ValueType>
class VolumeBoundary : public BoundaryPatchMixin
{
public:

    VolumeBoundary(const UnstructuredMesh& mesh, const Dictionary dict, int patchID)
        : BoundaryPatchMixin(
            mesh.boundaryMesh().offset()[patchID],
            mesh.boundaryMesh().offset()[patchID + 1],
            patchID
        ),
          boundaryCorrectionStrategy_(VolumeBoundaryFactory<ValueType>::create(mesh, dict, patchID))
    {}

    virtual void correctBoundaryConditions(DomainField<ValueType>& domainField)
    {
        boundaryCorrectionStrategy_->correctBoundaryCondition(domainField);
    }

private:

    // NOTE needs full namespace to be not ambiguous
    std::unique_ptr<NeoFOAM::finiteVolume::cellCentred::VolumeBoundaryFactory<ValueType>>
        boundaryCorrectionStrategy_;
};

}
