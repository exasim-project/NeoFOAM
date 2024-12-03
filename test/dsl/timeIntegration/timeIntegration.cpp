// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "../common.hpp"

#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/timeIntegration/forwardEuler.hpp"
#include "NeoFOAM/dsl/solver.hpp"
#include "NeoFOAM/dsl/ddt.hpp"

TEST_CASE("TimeIntegration")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    auto mesh = NeoFOAM::createSingleCellMesh(exec);

    NeoFOAM::Dictionary fvSchemes;
    NeoFOAM::Dictionary ddtSchemes;
    ddtSchemes.insert("type", std::string("forwardEuler"));
    fvSchemes.insert("ddtSchemes", ddtSchemes);
    NeoFOAM::Dictionary fvSolution;

    Field fA(exec, 1, 2.0);
    BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

    std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
    auto vf = VolumeField(exec,"vf", mesh, fA, bf, bcs);

    SECTION("Create expression and perform explicitOperation on " + execName)
    {
        auto dummy = Dummy(vf);
        Operator ddtOperator = NeoFOAM::dsl::temporal::ddt(vf);

        // ddt(U) = f
        auto eqn = ddtOperator + dummy;
        double dt {2.0};

        // int(ddt(U)) + f = 0
        // (U^1-U^0)/dt = -f
        // U^1 = - f * dt + U^0, where dt = 2, f=1, U^0=2.0 -> U^1=-2.0
        solve(eqn, vf, dt, fvSchemes, fvSolution);
        REQUIRE(getField(vf.internalField()) == -2.0);
    }
}
