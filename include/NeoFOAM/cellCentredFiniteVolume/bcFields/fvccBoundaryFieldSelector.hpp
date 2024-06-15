// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <memory>
#include <functional>
#include <string>

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/fields.hpp"

#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"


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
