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
        REQUIRE(mesh.points()[0][0] == 0.0);
        REQUIRE(mesh.points()[nCells][0] == 1.0);
        REQUIRE(mesh.points()[1][0] == 0.25);
        REQUIRE(mesh.points()[2][0] == 0.5);
        REQUIRE(mesh.points()[3][0] == 0.75);

        // Verify cell centers
        REQUIRE(mesh.cellCentres()[0][0] == 0.125);
        REQUIRE(mesh.cellCentres()[1][0] == 0.375);
        REQUIRE(mesh.cellCentres()[2][0] == 0.625);
        REQUIRE(mesh.cellCentres()[3][0] == 0.875);

        // Verify face owners
        REQUIRE(mesh.faceOwner()[0] == 0);
        REQUIRE(mesh.faceOwner()[1] == 0);
        REQUIRE(mesh.faceOwner()[2] == 1);
        REQUIRE(mesh.faceOwner()[3] == 2);
        REQUIRE(mesh.faceOwner()[4] == 3);

        // Verify face neighbors
        REQUIRE(mesh.faceNeighbour()[1] == 1);
        REQUIRE(mesh.faceNeighbour()[2] == 2);
        REQUIRE(mesh.faceNeighbour()[3] == 3);

        // Verify boundary mesh
        REQUIRE(mesh.boundaryMesh().faceCells()[0] == 0);
        REQUIRE(mesh.boundaryMesh().faceCells()[1] == 3);
        REQUIRE(mesh.boundaryMesh().cn()[0][0] == 0.125);
        REQUIRE(mesh.boundaryMesh().cn()[1][0] == 0.875);
        REQUIRE(mesh.boundaryMesh().delta()[0][0] == -0.125);
        REQUIRE(mesh.boundaryMesh().delta()[1][0] == 0.125);
    }
}
