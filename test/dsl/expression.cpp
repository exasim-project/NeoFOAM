// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "common.hpp"
#include "NeoFOAM/NeoFOAM.hpp"

using Expression = NeoFOAM::dsl::Expression;

TEST_CASE("Expression")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    auto mesh = NeoFOAM::createSingleCellMesh(exec);

    const size_t size {1};
    Field fA(exec, size, 2.0);
    BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

    std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
    auto vf = VolumeField(exec, "vf", mesh, fA, bf, bcs);
    auto fB = Field(exec, 1, 4.0);

    auto a = Dummy(vf);
    auto b = Dummy(vf);

    SECTION("Create equation and perform explicitOperation on " + execName)
    {
        auto eqnA = a + b;
        auto eqnB = fB * Dummy(vf) + 2 * Dummy(vf);
        auto eqnC = Expression(2 * a - b);
        auto eqnD = 3 * (2 * a - b);
        auto eqnE = (2 * a - b) + (2 * a - b);
        auto eqnF = (2 * a - b) - (2 * a - b);

        REQUIRE(eqnA.size() == 2);
        REQUIRE(eqnB.size() == 2);
        REQUIRE(eqnC.size() == 2);

        REQUIRE(getField(eqnA.explicitOperation(size)) == 4);
        REQUIRE(getField(eqnB.explicitOperation(size)) == 12);
        REQUIRE(getField(eqnC.explicitOperation(size)) == 2);
        REQUIRE(getField(eqnD.explicitOperation(size)) == 6);
        REQUIRE(getField(eqnE.explicitOperation(size)) == 4);
        REQUIRE(getField(eqnF.explicitOperation(size)) == 0);
    }
}
