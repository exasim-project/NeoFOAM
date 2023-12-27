// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <array>
#include <vector>

#include "NeoFOAM/blas/fields.hpp"

namespace NeoFOAM
{

class Stencil<int8_t>; // 0 is dynamic

    struct unstructuredMesh
    {

        unstructuredMesh(NeoFOAM::vectorField Sf, NeoFOAM::labelField owner, NeoFOAM::labelField neighbour, NeoFOAM::scalarField V, int32_t nCells, int32_t nInternalFaces)
            : Sf_(Sf)
            , owner_(owner)
            , neighbour_(neighbour)
            , V_(V)
            , nCells_(nCells)
            , nInternalFaces_(nInternalFaces){

            };

        vectorField Sf_; // area vector

        labelField owner_;     // owner cell
        labelField neighbour_; // neighbour cell

        scalarField V_; // cell volume

        int32_t nCells_; // number of cells
        int32_t nInternalFaces_; // number of internal faces
    };

} // namespace NeoFOAM