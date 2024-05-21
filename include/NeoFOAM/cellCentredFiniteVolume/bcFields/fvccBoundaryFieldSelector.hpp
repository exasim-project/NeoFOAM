// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <memory>
#include <functional>
#include <string>

#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/mesh/unstructured/UnstructuredMesh.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "NeoFOAM/core/Dictionary.hpp"

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarFixedValueBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarZeroGradientBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarEmptyBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarCalculatedBoundaryField.hpp"

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorFixedValueBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorZeroGradientBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorEmptyBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorCalculatedBoundaryField.hpp"

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccSurfaceBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/surface/scalar/fvccSurfaceScalarCalculatedBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/surface/scalar/fvccSurfaceScalarEmptyBoundaryField.hpp"


namespace NeoFOAM
{

template<typename T>
std::unique_ptr<fvccBoundaryField<T>>
getBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict)
{}

template<>
std::unique_ptr<fvccBoundaryField<scalar>>
getBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict);

template<>
std::unique_ptr<fvccBoundaryField<Vector>>
getBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict);


template<typename T>
std::unique_ptr<fvccSurfaceBoundaryField<T>>
getSurfaceBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict)
{}

template<>
std::unique_ptr<fvccSurfaceBoundaryField<scalar>>
getSurfaceBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict);

template<>
std::unique_ptr<fvccSurfaceBoundaryField<Vector>>
getSurfaceBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict);


template<typename T>
std::vector<std::unique_ptr<fvccSurfaceBoundaryField<T>>>
createCalculatedBCs(const UnstructuredMesh& mesh)
{}

template<>
std::vector<std::unique_ptr<fvccSurfaceBoundaryField<scalar>>>
createCalculatedBCs(const UnstructuredMesh& mesh);

template<>
std::vector<std::unique_ptr<fvccSurfaceBoundaryField<Vector>>>
createCalculatedBCs(const UnstructuredMesh& mesh);

}; // namespace NeoFOAM
