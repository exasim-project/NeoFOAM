// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"

namespace NeoFOAM
{

unstructuredMesh::unstructuredMesh(
    vectorField points,
    scalarField cellVolumes,
    vectorField cellCentres,
    vectorField faceAreas,
    vectorField faceCentres,
    scalarField magFaceAreas,
    labelField faceOwner,
    labelField faceNeighbour,
    int32_t nCells,
    int32_t nInternalFaces
)
    : points_(points),
      cellVolumes_(cellVolumes),
      cellCentres_(cellCentres),
      faceAreas_(faceAreas),
      faceCentres_(faceCentres),
      magFaceAreas_(magFaceAreas),
      faceOwner_(faceOwner),
      faceNeighbour_(faceNeighbour),
      nCells_(nCells),
      nInternalFaces_(nInternalFaces) {

      };

const vectorField& unstructuredMesh::points() const
{
    return points_;
}

const vectorField& unstructuredMesh::cellCentres() const
{
    return cellCentres_;
}

const scalarField& unstructuredMesh::cellVolumes() const
{
    return cellVolumes_;
}

const vectorField& unstructuredMesh::faceCentres() const
{
    return faceCentres_;
}

const vectorField& unstructuredMesh::faceAreas() const
{
    return faceAreas_;
}

const scalarField& unstructuredMesh::magFaceAreas() const
{
    return magFaceAreas_;
}

const labelField& unstructuredMesh::faceOwner() const
{
    return faceOwner_;
}

const labelField& unstructuredMesh::faceNeighbour() const
{
    return faceNeighbour_;
}

int32_t unstructuredMesh::nCells() const
{
    return nCells_;
}

int32_t unstructuredMesh::nInternalFaces() const
{
    return nInternalFaces_;
}

} // namespace NeoFOAM