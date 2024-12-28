// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

// free functions

namespace NeoFOAM::finiteVolume::cellCentred
{

// FIXME this should be merged with other boundary.hpp
/* @brief creates a vector of boundary conditions of type calculated for every boundary
 *
 * @tparam Type of the Boundary ie SurfaceBoundary<scalar>
 */
template<typename BoundaryType>
std::vector<BoundaryType> createCalculatedBCs(const UnstructuredMesh& mesh)
{
    std::vector<BoundaryType> bcs;
    for (size_t patchID = 0; patchID < mesh.nBoundaries(); patchID++)
    {
        Dictionary patchDict({{"type", std::string("calculated")}});
        bcs.push_back(BoundaryType(mesh, patchDict, patchID));
    }
    return bcs;
};

}
