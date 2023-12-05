// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/blas/fields.hpp"



namespace NeoFOAM
{

struct unstructuredMesh
{

    vectorField Sf_;  // area vector

    labelField owner_;  // owner cell
    labelField neighbour_;  // neighbour cell

    scalarField V_;  // cell volume

    int32_t nCells_;  // number of cells
    
    int32_t nInternalFaces_;  // number of internal faces


};

}  // namespace NeoFOAM