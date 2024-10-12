// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "../common.hpp"

#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/dsl/equation.hpp"
#include "NeoFOAM/dsl/timeIntegration/ddt.hpp"


TEST_CASE("TimeIntegration")
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


    NeoFOAM::Dictionary dict, subDict;
    subDict.insert("type", std::string("forwardEuler"));
    dict.insert("ddtSchemes", subDict);

    auto dummy = Dummy(exec, vf);

    SECTION("Create equation and perform explicitOperation on " + execName)
    {
        Operator ddtOperator = NeoFOAM::dsl::temporal::Ddt(exec, vf);

        auto eqn = ddtOperator + dummy;

        eqn.solve(vf, dict);
    }

    // fvcc::TimeIntegration timeIntergrator(eqnSys, dict);
}
