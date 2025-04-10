// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


TEST_CASE("Unstructured Mesh")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Can create single cell mesh " + execName)
    {
        NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);

        REQUIRE(mesh.nCells() == 1);
        REQUIRE(mesh.nBoundaryFaces() == 4);
        REQUIRE(mesh.nInternalFaces() == 0);
        REQUIRE(mesh.nBoundaries() == 4);
    }

    SECTION("Can create domainField from mesh " + execName)
    {

        NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);
        NeoN::DomainField<NeoN::scalar> domainField(exec, mesh);

        REQUIRE(domainField.boundaryField().offset().size() == 5);
    }

    SECTION("Can create a 1D uniform mesh" + execName)
    {
        size_t nCells = 4;

        NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, nCells);

        REQUIRE(mesh.nCells() == 4);
        REQUIRE(mesh.nBoundaryFaces() == 2);
        REQUIRE(mesh.nInternalFaces() == 3);
        REQUIRE(mesh.nBoundaries() == 2);
        REQUIRE(mesh.nFaces() == 5);

        // Verify mesh points
        // bc  [   internal  ]  bc
        // 0.0 [ 0.25 | 0.50 ] 1.0
        auto hostPoints = mesh.points().copyToHost();
        REQUIRE(hostPoints.span()[0][0] == 0.25);
        REQUIRE(hostPoints.span()[1][0] == 0.5);
        REQUIRE(hostPoints.span()[3][0] == 0.0);
        REQUIRE(hostPoints.span()[nCells][0] == 1.0);

        // Verify cell centers
        auto hostCellCentres = mesh.cellCentres().copyToHost();
        REQUIRE(hostCellCentres.span()[0][0] == 0.125);
        REQUIRE(hostCellCentres.span()[1][0] == 0.375);
        REQUIRE(hostCellCentres.span()[2][0] == 0.625);
        REQUIRE(hostCellCentres.span()[3][0] == 0.875);

        // Verify face owners
        // |_3 0 |_0 1 |_1 2 |_2 3 |_4
        auto hostFaceOwner = mesh.faceOwner().copyToHost();
        REQUIRE(hostFaceOwner.span()[0] == 0);
        REQUIRE(hostFaceOwner.span()[1] == 1);
        REQUIRE(hostFaceOwner.span()[2] == 2);
        REQUIRE(hostFaceOwner.span()[3] == 0);
        REQUIRE(hostFaceOwner.span()[4] == 3);

        // Verify face neighbors
        // |_3 0 |_0 1 |_1 2 |_2 3 |_4
        auto hostFaceNeighbour = mesh.faceNeighbour().copyToHost();
        REQUIRE(hostFaceNeighbour.span()[0] == 1);
        REQUIRE(hostFaceNeighbour.span()[1] == 2);
        REQUIRE(hostFaceNeighbour.span()[2] == 3);

        // Verify boundary mesh
        auto hostBoundaryFaceCells = mesh.boundaryMesh().faceCells().copyToHost();
        auto hostBoundaryCn = mesh.boundaryMesh().cn().copyToHost();
        auto hostBoundaryDelta = mesh.boundaryMesh().delta().copyToHost();
        REQUIRE(hostBoundaryFaceCells.span()[0] == 0);
        REQUIRE(hostBoundaryFaceCells.span()[1] == 3);
        REQUIRE(hostBoundaryCn.span()[0][0] == 0.125);
        REQUIRE(hostBoundaryCn.span()[1][0] == 0.875);
        REQUIRE(hostBoundaryDelta.span()[0][0] == -0.125);
        REQUIRE(hostBoundaryDelta.span()[1][0] == 0.125);
    }
}
