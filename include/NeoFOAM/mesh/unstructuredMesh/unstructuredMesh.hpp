// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <array>
#include <vector>

#include "NeoFOAM/primitives/vector.hpp"
#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/BoundaryMesh.hpp"
#include "NeoFOAM/mesh/stencil/StencilDataBase.hpp"

namespace NeoFOAM
{

class unstructuredMesh
{
public:

    unstructuredMesh(
        vectorField points,
        scalarField cellVolumes,
        vectorField cellCentres,
        vectorField faceAreas,
        vectorField faceCentres,
        scalarField magFaceAreas,
        labelField faceOwner,
        labelField faceNeighbour,
        int32_t nCells,
        int32_t nInternalFaces,
        int32_t nBoundaryFaces,
        int32_t nBoundaries,
        int32_t nFaces,
        BoundaryMesh boundaryMesh
    );

    const vectorField& points() const;

    const scalarField& cellVolumes() const;

    const vectorField& cellCentres() const;

    const vectorField& faceCentres() const;

    const vectorField& faceAreas() const;

    const scalarField& magFaceAreas() const;

    const labelField& faceOwner() const;

    const labelField& faceNeighbour() const;

    label nCells() const;

    int32_t nInternalFaces() const;

    int32_t nBoundaryFaces() const;

    int32_t nBoundaries() const;

    int32_t nFaces() const;

    const BoundaryMesh& boundaryMesh() const;

    StencilDataBase& stencilDB() const;

    const executor& exec() const;

private:

    const executor exec_;

    vectorField points_; // points

    scalarField cellVolumes_; // cell volume
    vectorField cellCentres_; // cell centre

    vectorField faceAreas_;    // face area vector
    vectorField faceCentres_;  // face centre vector
    scalarField magFaceAreas_; // face area

    labelField faceOwner_;     // owner cell
    labelField faceNeighbour_; // neighbour cell

    int32_t nCells_;         // number of cells
    int32_t nInternalFaces_; // number of internal faces
    int32_t nBoundaryFaces_; // number of boundary faces
    int32_t nBoundaries_;    // number of boundaries
    int32_t nFaces_;         // number of faces


    BoundaryMesh boundaryMesh_; // boundary mesh

    mutable StencilDataBase stencilDataBase_;
};

} // namespace NeoFOAM
