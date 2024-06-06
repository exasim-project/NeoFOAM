// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <memory>

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/boundaryField.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/fields/fieldTypeDefs.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
std::unique_ptr<BoundaryField<ValueType>>
getBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict)
{}

template<>
std::unique_ptr<BoundaryField<scalar>>
getBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict);

template<>
std::unique_ptr<BoundaryField<Vector>>
getBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict);


template<typename ValueType>
std::unique_ptr<SurfaceBoundaryField<ValueType>>
getSurfaceBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict)
{}

template<>
std::unique_ptr<SurfaceBoundaryField<scalar>>
getSurfaceBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict);

template<>
std::unique_ptr<SurfaceBoundaryField<Vector>>
getSurfaceBC(const UnstructuredMesh& mesh, int patchID, const Dictionary& patchDict);


template<typename ValueType>
std::vector<std::unique_ptr<SurfaceBoundaryField<ValueType>>>
createCalculatedBCs(const UnstructuredMesh& mesh)
{}

template<>
std::vector<std::unique_ptr<SurfaceBoundaryField<scalar>>>
createCalculatedBCs(const UnstructuredMesh& mesh);

template<>
std::vector<std::unique_ptr<SurfaceBoundaryField<Vector>>>
createCalculatedBCs(const UnstructuredMesh& mesh);

}; // namespace NeoFOAM
