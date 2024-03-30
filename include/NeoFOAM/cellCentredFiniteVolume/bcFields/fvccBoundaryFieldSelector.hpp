// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <memory>
#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "NeoFOAM/core/Dictionary.hpp"

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarFixedValueBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarZeroGradientBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarEmptyBoundaryField.hpp"

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorFixedValueBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorZeroGradientBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorEmptyBoundaryField.hpp"

#include <functional>
#include <string>

namespace NeoFOAM
{

template <typename T>
std::unique_ptr<fvccBoundaryField<T>> getBC(const unstructuredMesh& mesh, int patchID, const Dictionary& patchDict)
{

}

template <>
std::unique_ptr<fvccBoundaryField<scalar>> getBC(const unstructuredMesh& mesh, int patchID, const Dictionary& patchDict)
{
    std::string type = patchDict.get<std::string>("type");
    if (type == "zeroGradient")
    {
        return std::make_unique<fvccScalarZeroGradientBoundaryField>(mesh, patchID);
    }
    else if (type == "fixedValue")
    {
        return std::make_unique<fvccScalarFixedValueBoundaryField>(mesh, patchID, patchDict.get<scalar>("value"));
    }
    else if (type == "empty")
    {
        return std::make_unique<fvccScalarEmptyBoundaryField>(mesh, patchID);
    }
    else
    {
        std::cout << "keyword not found" << std::endl;
    }
};

template <>
std::unique_ptr<fvccBoundaryField<Vector>> getBC(const unstructuredMesh& mesh, int patchID, const Dictionary& patchDict)
{
    std::string type = patchDict.get<std::string>("type");
    if (type == "zeroGradient")
    {
        return std::make_unique<fvccVectorZeroGradientBoundaryField>(mesh, patchID);
    }
    else if (type == "fixedValue")
    {
        return std::make_unique<fvccVectorFixedValueBoundaryField>(mesh, patchID, patchDict.get<Vector>("value"));
    }
    else if (type == "empty")
    {
        return std::make_unique<fvccVectorEmptyBoundaryField>(mesh, patchID);
    }
    else
    {
        std::cout << "keyword not found" << std::endl;
    }
};


}; // namespace NeoFOAM