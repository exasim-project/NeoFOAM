// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/fields/domainField.hpp"

TEST_CASE("Unstructured Mesh")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

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
        auto hostPoints = mesh.points().copyToHost();
        REQUIRE(hostPoints[0][0] == 0.0);
        REQUIRE(hostPoints[nCells][0] == 1.0);
        REQUIRE(hostPoints[1][0] == 0.25);
        REQUIRE(hostPoints[2][0] == 0.5);
        REQUIRE(hostPoints[3][0] == 0.75);

        // Verify cell centers
        auto hostCellCentres = mesh.cellCentres().copyToHost();
        REQUIRE(hostCellCentres[0][0] == 0.125);
        REQUIRE(hostCellCentres[1][0] == 0.375);
        REQUIRE(hostCellCentres[2][0] == 0.625);
        REQUIRE(hostCellCentres[3][0] == 0.875);

        // Verify face owners
        auto hostFaceOwner = mesh.faceOwner().copyToHost();
        REQUIRE(hostFaceOwner[0] == 0);
        REQUIRE(hostFaceOwner[1] == 0);
        REQUIRE(hostFaceOwner[2] == 1);
        REQUIRE(hostFaceOwner[3] == 2);
        REQUIRE(hostFaceOwner[4] == 3);

        // Verify face neighbors
        auto hostFaceNeighbour = mesh.faceNeighbour().copyToHost();
        REQUIRE(hostFaceNeighbour[1] == 1);
        REQUIRE(hostFaceNeighbour[2] == 2);
        REQUIRE(hostFaceNeighbour[3] == 3);

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
}
