// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <array>
#include <vector>

#include "NeoFOAM/blas/field.hpp"

namespace NeoFOAM
{

    struct unstructuredMesh
    {

        unstructuredMesh(NeoFOAM::vectorField Sf, NeoFOAM::labelField owner, NeoFOAM::labelField neighbour, NeoFOAM::scalarField V, int32_t nCells, int32_t nInternalFaces);

        vectorField Sf_; // area vector

        labelField owner_;     // owner cell
        labelField neighbour_; // neighbour cell

        scalarField V_; // cell volume

        int32_t nCells_; // number of cells
        int32_t nInternalFaces_; // number of internal faces

        int32_t nBoundaryFaces_; // number of faces
        int32_t nBoundaryPatches_; // number of patches

        std::vector<int32_t> boundaryStartFace_; // start face of boundary patch
    };

} // namespace NeoFOAM