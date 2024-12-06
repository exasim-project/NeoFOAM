// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

#include "NeoFOAM/core/primitives/vector.hpp" // for Vector


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
        faceAreasVectors,                                                               // nf,
        {exec, {{0.5, 0.0, 0.0}, {0.0, -0.5, 0.0}, {-0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}}}, // delta
        {exec, {1, 1, 1, 1}},                                                           // weights
        {exec, {2.0, 2.0, 2.0, 2.0}}, // deltaCoeffs --> mag(1 / delta)
        {0, 1, 2, 3, 4}               // offset
    );
    return UnstructuredMesh(
        {exec, {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}}, // points,
        {exec, 1},                                            // cellVolumes
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

UnstructuredMesh create1DUniformMesh(const Executor exec, const size_t nCells)
{
    const Vector leftBoundary = {0.0, 0.0, 0.0};
    const Vector rightBoundary = {1.0, 0.0, 0.0};
    scalar meshSpacing = (rightBoundary[0] - leftBoundary[0]) / static_cast<scalar>(nCells);
    auto hostExec = SerialExecutor {};
    vectorField meshPointsHost(hostExec, nCells + 1, {0.0, 0.0, 0.0});
    meshPointsHost[0] = leftBoundary;
    meshPointsHost[nCells] = rightBoundary;
    auto meshPoints = meshPointsHost.copyToExecutor(exec);

    // loop over internal mesh points
    auto meshPointsSpan = meshPoints.span();
    auto leftBoundaryX = leftBoundary[0];
    parallelFor(
        exec,
        {1, nCells},
        KOKKOS_LAMBDA(const size_t i) {
            meshPointsSpan[i][0] = leftBoundaryX + static_cast<scalar>(i) * meshSpacing;
        }
    );

    scalarField cellVolumes(exec, nCells, meshSpacing);

    vectorField cellCenters(exec, nCells, {0.0, 0.0, 0.0});
    auto cellCentersSpan = cellCenters.span();
    parallelFor(
        exec,
        {0, nCells},
        KOKKOS_LAMBDA(const size_t i) {
            cellCentersSpan[i][0] = 0.5 * (meshPointsSpan[i][0] + meshPointsSpan[i + 1][0]);
        }
    );


    vectorField faceAreasHost(hostExec, nCells + 1, {1.0, 0.0, 0.0});
    faceAreasHost[0] = {-1.0, 0.0, 0.0}; // left boundary face
    auto faceAreas = faceAreasHost.copyToExecutor(exec);

    vectorField faceCenters(exec, meshPoints);
    scalarField magFaceAreas(exec, nCells + 1, 1.0);

    labelField faceOwnerHost(hostExec, nCells + 1);
    faceOwnerHost[0] = 0; // left boundary face
    auto faceOwner = faceOwnerHost.copyToExecutor(exec);

    // loop over internal faces and right boundary face
    auto faceOwnerSpan = faceOwner.span();
    parallelFor(
        exec,
        {1, nCells + 1},
        KOKKOS_LAMBDA(const size_t i) { faceOwnerSpan[i] = static_cast<label>(i - 1); }
    );

    labelField faceNeighbor(exec, nCells + 1);

    auto faceNeighborSpan = faceNeighbor.span();
    parallelFor(
        exec,
        {0, nCells},
        KOKKOS_LAMBDA(const size_t i) { faceNeighborSpan[i] = static_cast<label>(i); }
    );

    vectorField deltaHost(hostExec, 2);
    auto cellCentersHost = cellCenters.copyToHost();
    deltaHost[0] = {leftBoundary[0] - cellCentersHost[0][0], 0.0, 0.0};
    deltaHost[1] = {rightBoundary[0] - cellCentersHost[nCells - 1][0], 0.0, 0.0};
    auto delta = deltaHost.copyToExecutor(exec);

    scalarField deltaCoeffsHost(hostExec, 2);
    deltaCoeffsHost[0] = 1 / mag(deltaHost[0]);
    deltaCoeffsHost[1] = 1 / mag(deltaHost[1]);
    auto deltaCoeffs = deltaCoeffsHost.copyToExecutor(exec);

    BoundaryMesh boundaryMesh(
        exec,
        {exec, {0, static_cast<int>(nCells) - 1}},
        {exec, {leftBoundary, rightBoundary}},
        {exec, {cellCentersHost[0], cellCentersHost[nCells - 1]}},
        {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
        {exec, {1.0, 1.0}},
        {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
        delta,
        {exec, {1.0, 1.0}},
        deltaCoeffs,
        {0, 1, 2}
    );

    return UnstructuredMesh(
        meshPoints,
        cellVolumes,
        cellCenters,
        faceAreas,
        faceCenters,
        magFaceAreas,
        faceOwner,
        faceNeighbor,
        nCells,
        nCells - 1,
        2,
        2,
        nCells + 1,
        boundaryMesh
    );
}
} // namespace NeoFOAM
