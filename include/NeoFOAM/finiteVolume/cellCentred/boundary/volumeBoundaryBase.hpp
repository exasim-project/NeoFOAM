// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/fieldTypeDefs.hpp"
#include "NeoFOAM/fields/boundaryFields.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/boundaryBase.hpp"
#include "NeoFOAM/core/registerClass.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/core/dictionary.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{
// forward declaration so we can use it to define the create function and the class manager
template<typename ValueType>
class VolumeBoundaryModel;

// define the create function use to instantiate the derived classes
template<typename ValueType>
using createFunc = std::function<std::unique_ptr<VolumeBoundaryModel<ValueType>>(
    const UnstructuredMesh& mesh, const Dictionary dict, int patchID
)>;

// define the class manager to register the classes
template<typename ValueType>
using VolumeBoundaryModelManager =
    NeoFOAM::BaseClassRegistry<VolumeBoundaryModel<ValueType>, createFunc<ValueType>>;

template<typename ValueType>
class VolumeBoundaryModel : public VolumeBoundaryModelManager<ValueType>
{
public:

    template<typename derivedClass>
    using VolumeBoundaryModelReg = NeoFOAM::
        RegisteredClass<derivedClass, VolumeBoundaryModel<ValueType>, createFunc<ValueType>>;

    template<typename derivedClass>
    bool registerClass()
    {
        return VolumeBoundaryModel<ValueType>::template VolumeBoundaryModelReg<derivedClass>::reg;
    }

    static std::unique_ptr<VolumeBoundaryModel<ValueType>> create(
        const std::string& name, const UnstructuredMesh& mesh, const Dictionary& dict, int patchID
    )
    {
        try
        {
            auto func = VolumeBoundaryModelManager<ValueType>::classMap().at(name);
            return func(mesh, dict, patchID);
        }
        catch (const std::out_of_range& e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return nullptr;
        }
    }

    virtual void correctBoundaryConditions(DomainField<ValueType>& domainField) = 0;
};


/**
 * @brief Represents a surface boundary field for a cell-centered finite volume method.
 *
 * @tparam ValueType The data type of the field.
 */
template<typename ValueType>
class VolumeBoundary : public BoundaryBase<ValueType>
{
public:

    VolumeBoundary(const UnstructuredMesh& mesh, const Dictionary dict, int patchID)
        : BoundaryBase<ValueType>(mesh, patchID),
          bcModel_(VolumeBoundaryModel<ValueType>::create(mesh, dict, patchID))
    {}

    virtual void correctBoundaryConditions(DomainField<ValueType>& domainField)
    {
        bcModel_->correctBoundaryConditions(domainField);
    }

private:

    std::unique_ptr<VolumeBoundaryModel<ValueType>> bcModel_;
};

}
