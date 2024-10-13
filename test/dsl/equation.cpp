// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "common.hpp"
#include "NeoFOAM/dsl/equation.hpp"

using Equation = NeoFOAM::dsl::Equation;

TEST_CASE("Equation")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    auto mesh = NeoFOAM::createSingleCellMesh(exec);

    Field fA(exec, 1, 2.0);
    BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

    std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
    auto vf = VolumeField(exec, mesh, fA, bf, bcs);
    auto fB = Field(exec, 1, 4.0);

    auto a = Dummy(exec, vf);
    auto b = Dummy(exec, vf);

    SECTION("Create equation and perform explicitOperation on " + execName)
    {
        auto eqnA = a + b;
        auto eqnB = fB * Dummy(exec, vf) + 2 * Dummy(exec, vf);
        auto eqnC = Equation(2 * a - b);
        auto eqnD = 3 * (2 * a - b);
        auto eqnE = (2 * a - b) + (2 * a - b);
        auto eqnF = (2 * a - b) - (2 * a - b);

        REQUIRE(eqnA.size() == 2);
        REQUIRE(eqnB.size() == 2);
        REQUIRE(eqnC.size() == 2);

        REQUIRE(getField(eqnA.explicitOperation()) == 4);
        REQUIRE(getField(eqnB.explicitOperation()) == 12);
        REQUIRE(getField(eqnC.explicitOperation()) == 2);
        REQUIRE(getField(eqnD.explicitOperation()) == 6);
        REQUIRE(getField(eqnE.explicitOperation()) == 4);
        REQUIRE(getField(eqnF.explicitOperation()) == 0);
    }
}
