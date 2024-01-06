// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"

namespace NeoFOAM
{

    unstructuredMesh::unstructuredMesh(NeoFOAM::vectorField Sf, NeoFOAM::labelField owner, NeoFOAM::labelField neighbour, NeoFOAM::scalarField V, int32_t nCells, int32_t nInternalFaces)
        : Sf_(Sf)
        , owner_(owner)
        , neighbour_(neighbour)
        , V_(V)
        , nCells_(nCells)
        , nInternalFaces_(nInternalFaces){

        };


} // namespace NeoFOAM