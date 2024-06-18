// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/core/registerClass.hpp"
#include "NeoFOAM/fields.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// forward declaration so we can use it to define the create function and the class manager
template<typename ValueType>
class VolumeBoundaryModel;

// a detail namespace to prevent conflicts for surface boundaries
namespace cellCentred::VolummeBoundarDetail
{

    // define the create function use to instantiate the derived classes
    template<typename ValueType>
    using CreateFunc = std::function<std::unique_ptr<VolumeBoundaryModel<ValueType>>(
        const UnstructuredMesh&, const Dictionary, int
    )>;

    template<typename ValueType>
    using ClassRegistry =
        NeoFOAM::BaseClassRegistry<VolumeBoundaryModel<ValueType>, CreateFunc<ValueType>>;
}

using namespace cellCentred::VolummeBoundarDetail;

template<typename ValueType>
class VolumeBoundaryModel : public ClassRegistry<ValueType>
{

private:

    template<typename DerivedClass>
    using VolumeBoundaryModelReg = NeoFOAM::
        RegisteredClass<DerivedClass, VolumeBoundaryModel<ValueType>, CreateFunc<ValueType>>;

public:

    template<typename DerivedClass>
    bool registerClass() const
    {
        return VolumeBoundaryModel<ValueType>::template VolumeBoundaryModelReg<DerivedClass>::reg;
    }

    static std::unique_ptr<VolumeBoundaryModel<ValueType>> create(
        const std::string& name, const UnstructuredMesh& mesh, const Dictionary& dict, int patchID
    )
    {
        try
        {
            auto func = ClassRegistry<ValueType>::classMap().at(name);
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
          bcModel_(NeoFOAM::finiteVolume::cellCentred::VolumeBoundaryModel<ValueType>::create(
              mesh, dict, patchID
          ))
    {}

    virtual void correctBoundaryConditions(DomainField<ValueType>& domainField)
    {
        bcModel_->correctBoundaryConditions(domainField);
    }

private:

    // NOTE needs full namespace to be not ambiguous
    std::unique_ptr<NeoFOAM::finiteVolume::cellCentred::VolumeBoundaryModel<ValueType>> bcModel_;
};

}
