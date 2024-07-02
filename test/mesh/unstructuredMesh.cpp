// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

TEST_CASE("Unstructured Mesh")
{

    SECTION("Can create single cell mesh ")
    {
        NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh();

        REQUIRE(mesh.nCells() == 1);
    }
}
