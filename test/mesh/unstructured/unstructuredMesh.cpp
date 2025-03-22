// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/fields/domainField.hpp"
#include "executorGenerator.hpp"

TEST_CASE("Unstructured Mesh")
{
    NeoFOAM::Executor exec = GENERATE(allAvailableExecutor());

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("Can create single cell mesh " + execName)
    {
        NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);

        REQUIRE(mesh.nCells() == 1);
        REQUIRE(mesh.nBoundaryFaces() == 4);
        REQUIRE(mesh.nInternalFaces() == 0);
        REQUIRE(mesh.nBoundaries() == 4);
    }

    SECTION("Can create domainField from mesh " + execName)
    {

        NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, mesh);

        REQUIRE(domainField.boundaryField().offset().size() == 5);
    }

    SECTION("Can create a 1D uniform mesh" + execName)
    {
        size_t nCells = 4;

        NeoFOAM::UnstructuredMesh mesh = NeoFOAM::create1DUniformMesh(exec, nCells);

        REQUIRE(mesh.nCells() == 4);
        REQUIRE(mesh.nBoundaryFaces() == 2);
        REQUIRE(mesh.nInternalFaces() == 3);
        REQUIRE(mesh.nBoundaries() == 2);
        REQUIRE(mesh.nFaces() == 5);

        // Verify mesh points
        // bc  [   internal  ]  bc
        // 0.0 [ 0.25 | 0.50 ] 1.0
        auto hostPoints = mesh.points().copyToHost();
        REQUIRE(hostPoints[0][0] == 0.25);
        REQUIRE(hostPoints[1][0] == 0.5);
        REQUIRE(hostPoints[3][0] == 0.0);
        REQUIRE(hostPoints[nCells][0] == 1.0);

        // Verify cell centers
        auto hostCellCentres = mesh.cellCentres().copyToHost();
        REQUIRE(hostCellCentres[0][0] == 0.125);
        REQUIRE(hostCellCentres[1][0] == 0.375);
        REQUIRE(hostCellCentres[2][0] == 0.625);
        REQUIRE(hostCellCentres[3][0] == 0.875);

        // Verify face owners
        // |_3 0 |_0 1 |_1 2 |_2 3 |_4
        auto hostFaceOwner = mesh.faceOwner().copyToHost();
        REQUIRE(hostFaceOwner[0] == 0);
        REQUIRE(hostFaceOwner[1] == 1);
        REQUIRE(hostFaceOwner[2] == 2);
        REQUIRE(hostFaceOwner[3] == 0);
        REQUIRE(hostFaceOwner[4] == 3);

        // Verify face neighbors
        // |_3 0 |_0 1 |_1 2 |_2 3 |_4
        auto hostFaceNeighbour = mesh.faceNeighbour().copyToHost();
        REQUIRE(hostFaceNeighbour[0] == 1);
        REQUIRE(hostFaceNeighbour[1] == 2);
        REQUIRE(hostFaceNeighbour[2] == 3);

        // Verify boundary mesh
        auto hostBoundaryFaceCells = mesh.boundaryMesh().faceCells().copyToHost();
        auto hostBoundaryCn = mesh.boundaryMesh().cn().copyToHost();
        auto hostBoundaryDelta = mesh.boundaryMesh().delta().copyToHost();
        REQUIRE(hostBoundaryFaceCells[0] == 0);
        REQUIRE(hostBoundaryFaceCells[1] == 3);
        REQUIRE(hostBoundaryCn[0][0] == 0.125);
        REQUIRE(hostBoundaryCn[1][0] == 0.875);
        REQUIRE(hostBoundaryDelta[0][0] == -0.125);
        REQUIRE(hostBoundaryDelta[1][0] == 0.125);
    }

    SECTION("Can compute sparsityPattern from mesh " + execName)
    {
        size_t nCells = 4;
        size_t nnzs = nCells + (nCells - 1) * 2; // 2 nnz per internal face
        NeoFOAM::UnstructuredMesh mesh = NeoFOAM::create1DUniformMesh(exec, nCells);

        auto [rows, cols, map] = createSparsityPattern<NeoFOAM::localIdx>(exec, mesh);
        auto [rowsH, colsH, mapH] = copyToHosts(rows, cols, map);
        auto [rowsHS, colsHS, mapHS] = spans(rowsH, colsH, mapH);

        REQUIRE(rowsHS.size() == nnzs);
        REQUIRE(rowsHS[0] == 0);
        REQUIRE(rowsHS[1] == 0);
        REQUIRE(rowsHS[2] == 1);
        REQUIRE(rowsHS[3] == 1);
        REQUIRE(rowsHS[4] == 1);
        REQUIRE(rowsHS[5] == 2);
        REQUIRE(rowsHS[6] == 2);
        REQUIRE(rowsHS[7] == 2);
        REQUIRE(rowsHS[8] == 3);
        REQUIRE(rowsHS[9] == 3);

        REQUIRE(colsHS[0] == 0);
        REQUIRE(colsHS[1] == 1);
        REQUIRE(colsHS[2] == 0);
        REQUIRE(colsHS[3] == 1);
        REQUIRE(colsHS[4] == 2);
        REQUIRE(colsHS[5] == 1);
        REQUIRE(colsHS[6] == 2);
        REQUIRE(colsHS[7] == 3);
        REQUIRE(colsHS[8] == 2);
        REQUIRE(colsHS[9] == 3);

        REQUIRE(mapHS[0] == 6);
        REQUIRE(mapHS[1] == 0);
        REQUIRE(mapHS[2] == 1);
        REQUIRE(mapHS[3] == 7);
        REQUIRE(mapHS[4] == 2);
        REQUIRE(mapHS[5] == 3);
        REQUIRE(mapHS[6] == 8);
        REQUIRE(mapHS[7] == 4);
        REQUIRE(mapHS[8] == 5);
        REQUIRE(mapHS[9] == 9);
    }
}
