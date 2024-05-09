// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/unstructuredMesh/UnstructuredMesh.hpp"

namespace NeoFOAM
{

UnstructuredMesh::UnstructuredMesh(
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
)
    : exec_(points.exec()),
      points_(points),
      cellVolumes_(cellVolumes),
      cellCentres_(cellCentres),
      faceAreas_(faceAreas),
      faceCentres_(faceCentres),
      magFaceAreas_(magFaceAreas),
      faceOwner_(faceOwner),
      faceNeighbour_(faceNeighbour),
      nCells_(nCells),
      nInternalFaces_(nInternalFaces),
      nBoundaryFaces_(nBoundaryFaces),
      nBoundaries_(nBoundaries),
      nFaces_(nFaces),
      boundaryMesh_(boundaryMesh),
      stencilDataBase_() {

      };

const vectorField& UnstructuredMesh::points() const
{
    return points_;
}


const scalarField& UnstructuredMesh::cellVolumes() const
{
    return cellVolumes_;
}

const vectorField& UnstructuredMesh::cellCentres() const
{
    return cellCentres_;
}

const vectorField& UnstructuredMesh::faceCentres() const
{
    return faceCentres_;
}

const vectorField& UnstructuredMesh::faceAreas() const
{
    return faceAreas_;
}

const scalarField& UnstructuredMesh::magFaceAreas() const
{
    return magFaceAreas_;
}

const labelField& UnstructuredMesh::faceOwner() const
{
    return faceOwner_;
}

const labelField& UnstructuredMesh::faceNeighbour() const
{
    return faceNeighbour_;
}

int32_t UnstructuredMesh::nCells() const
{
    return nCells_;
}

int32_t UnstructuredMesh::nInternalFaces() const
{
    return nInternalFaces_;
}

int32_t UnstructuredMesh::nBoundaryFaces() const
{
    return nBoundaryFaces_;
}

int32_t UnstructuredMesh::nBoundaries() const
{
    return nBoundaries_;
}

int32_t UnstructuredMesh::nFaces() const
{
    return nFaces_;
}

const BoundaryMesh& UnstructuredMesh::boundaryMesh() const
{
    return boundaryMesh_;
}

StencilDataBase& UnstructuredMesh::stencilDB() const
{
    return stencilDataBase_;
}

const executor& UnstructuredMesh::exec() const
{
    return exec_;
}

} // namespace NeoFOAM
