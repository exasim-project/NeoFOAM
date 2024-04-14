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
      stencilDataBase_()
      {

      };

const vectorField& unstructuredMesh::points() const
{
    return points_;
}


const scalarField& unstructuredMesh::cellVolumes() const
{
    return cellVolumes_;
}

const vectorField& unstructuredMesh::cellCentres() const
{
    return cellCentres_;
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

int32_t unstructuredMesh::nBoundaryFaces() const
{
    return nBoundaryFaces_;
}

int32_t unstructuredMesh::nBoundaries() const
{
    return nBoundaries_;
}

int32_t unstructuredMesh::nFaces() const
{
    return nFaces_;
}

const BoundaryMesh& unstructuredMesh::boundaryMesh() const
{
    return boundaryMesh_;
}

StencilDataBase& unstructuredMesh::stencilDB()
{
    return stencilDataBase_;
}

const executor& unstructuredMesh::exec() const
{
    return exec_;
}

} // namespace NeoFOAM