// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/boundaryBase.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// forward declaration so we can use it to define the create function and the class manager
template<typename ValueType>
class BoundaryCorrectionStrategy;

// define the create function use to instantiate the derived classes
template<typename ValueType>
using CreateFunc = std::function<std::unique_ptr<BoundaryCorrectionStrategy<ValueType>>(
    std::shared_ptr<const UnstructuredMesh>, const Dictionary, int
)>;

// define the class manager to register the classes
template<typename ValueType>
using BoundaryRegistry =
    NeoFOAM::BaseClassRegistry<BoundaryCorrectionStrategy<ValueType>, CreateFunc<ValueType>>;

/* This class implements the 'Strategy' part of the Strategy pattern
**
**
*/
template<typename ValueType>
class BoundaryCorrectionStrategy
{
public:

    // /* Create a boundary condition for specific path by its type name
    //     *
    //     */
    // static std::unique_ptr<BoundaryCorrectionStrategy> createConcrete(
    //     const std::string& name,
    //     std::shared_ptr<const NeoFOAM::UnstructuredMesh> mesh,
    //     const NeoFOAM::Dictionary& dict,
    //     int patchID
    // )
    // {
    //     // TODO FIXME where are start and end are comming from?
    //     // we need a method that gets start and end for a given patchID
    //     // I guess it should be derived from boundaryFields.hpp
    //     auto start = dict.get<std::size_t>("start");
    //     auto end = dict.get<std::size_t>("end");
    //     return std::make_unique<BoundaryFactory>(
    //         name,
    //         start,
    //         end,
    //         patchID,
    //         dict);
    // }

    virtual void correctBoundaryConditionsImpl(DomainField<ValueType>& domainField) = 0;
};


template<typename ValueType, typename BoundaryType>
class BoundaryFactory;


/**
 * @brief Represents a boundary condition  for a cell-centered finite volume method.
 *
 * @tparam ValueType The data type of the field.
 * @tparam BoundaryType Whether the boundary is a volume or surface boundary
 */
template<typename ValueType, typename BoundaryType>
class BoundaryFactory :
    public BoundaryPatchMixin<ValueType>,
    public BaseClassRegistry<BoundaryFactory<ValueType, int>, CreateFunc<ValueType>>
{
public:

    MAKE_CLASS_A_RUNTIME_FACTORY();

    BoundaryFactory(std::shared_ptr<const UnstructuredMesh> mesh, int patchID)
        : BoundaryPatchMixin<ValueType>(mesh, patchID)
    {}

    void
    setCorrectionStrategy(std::unique_ptr<BoundaryCorrectionStrategy<ValueType>> correctionStrategy)
    {
        concreteBoundary_ = std::move(correctionStrategy);
    }

    virtual void correctBoundaryConditions(DomainField<ValueType>& domainField)
    {
        concreteBoundary_->correctBoundaryConditionsImpl(domainField);
    }


private:

    std::unique_ptr<BoundaryCorrectionStrategy<ValueType>> concreteBoundary_;
};

}
