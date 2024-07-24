// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

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
    size_t nCells,
    size_t nInternalFaces,
    size_t nBoundaryFaces,
    size_t nBoundaries,
    size_t nFaces,
    BoundaryMesh boundaryMesh
)
    : exec_(points.exec()), points_(points), cellVolumes_(cellVolumes), cellCentres_(cellCentres),
      faceAreas_(faceAreas), faceCentres_(faceCentres), magFaceAreas_(magFaceAreas),
      faceOwner_(faceOwner), faceNeighbour_(faceNeighbour), nCells_(nCells),
      nInternalFaces_(nInternalFaces), nBoundaryFaces_(nBoundaryFaces), nBoundaries_(nBoundaries),
      nFaces_(nFaces), boundaryMesh_(boundaryMesh), stencilDataBase_()
{}


const vectorField& UnstructuredMesh::points() const { return points_; }

const scalarField& UnstructuredMesh::cellVolumes() const { return cellVolumes_; }

const vectorField& UnstructuredMesh::cellCentres() const { return cellCentres_; }

const vectorField& UnstructuredMesh::faceCentres() const { return faceCentres_; }

const vectorField& UnstructuredMesh::faceAreas() const { return faceAreas_; }

const scalarField& UnstructuredMesh::magFaceAreas() const { return magFaceAreas_; }

const labelField& UnstructuredMesh::faceOwner() const { return faceOwner_; }

const labelField& UnstructuredMesh::faceNeighbour() const { return faceNeighbour_; }

size_t UnstructuredMesh::nCells() const { return nCells_; }

size_t UnstructuredMesh::nInternalFaces() const { return nInternalFaces_; }

size_t UnstructuredMesh::nBoundaryFaces() const { return nBoundaryFaces_; }

size_t UnstructuredMesh::nBoundaries() const { return nBoundaries_; }

size_t UnstructuredMesh::nFaces() const { return nFaces_; }

const BoundaryMesh& UnstructuredMesh::boundaryMesh() const { return boundaryMesh_; }

StencilDataBase& UnstructuredMesh::stencilDB() const { return stencilDataBase_; }

const Executor& UnstructuredMesh::exec() const { return exec_; }

UnstructuredMesh createSingleCellMesh(const Executor exec)
{
    // a 2D mesh in 3D space with left, right, top, bottom boundary faces
    // with the centre at (0.5, 0.5, 0.0)
    // left, top, right, bottom faces
    // and four boundaries one left, right, top, bottom

    vectorField faceAreasVectors(exec, {{-1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, -1, 0}});
    vectorField faceCentresVectors(
        exec, {{0.0, 0.5, 0.0}, {0.5, 1.0, 0.0}, {1.0, 0.5, 0.0}, {0.5, 0.0, 0.0}}
    );
    scalarField magFaceAreas(exec, {1, 1, 1, 1});

    BoundaryMesh boundaryMesh(
        exec,
        {exec, {0, 0, 0, 0}},                                                           // faceCells
        faceCentresVectors,                                                             // cf
        faceAreasVectors,                                                               // cn,
        faceAreasVectors,                                                               // sf,
        magFaceAreas,                                                                   // magSf,
        faceAreasVectors,                                                               //
        {exec, {{0.5, 0.0, 0.0}, {0.0, -0.5, 0.0}, {-0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}}}, // delta
        {exec, {1, 1, 1, 1}},                                                           // weights
        {exec, {2.0, 2.0, 2.0, 2.0}}, // deltaCoeffs --> mag(1 / delta)
        {0, 1, 2, 3, 4}               // offset
    );
    return UnstructuredMesh(
        {exec, {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}}, // points,
        {exec, {1}},                                          // cellVolumes
        {exec, {{0.5, 0.5, 0.0}}},                            // cellCentres
        faceAreasVectors,
        faceCentresVectors,
        magFaceAreas,
        {exec, {0, 0, 0, 0}}, // faceOwner
        {exec, {}},           // faceNeighbour,
        1,                    // nCells
        0,                    // nInternalFaces,
        4,                    // nBoundaryFaces,
        4,                    // nBoundaries,
        4,                    // nFaces,
        boundaryMesh
    );
}
} // namespace NeoFOAM
