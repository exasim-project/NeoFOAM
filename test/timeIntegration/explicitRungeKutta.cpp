// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "../dsl/common.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/dsl/expression.hpp"
#include "NeoFOAM/dsl/ddt.hpp"
#include "NeoFOAM/timeIntegration/explicitRungeKutta.hpp"

TEST_CASE("TimeIntegration")
{
    auto exec = NeoFOAM::SerialExecutor();

    // std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    auto mesh = NeoFOAM::createSingleCellMesh(exec);

    Field fA(exec, 1, 2.0);
    BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

    std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
    auto vf = VolumeField(exec, mesh, fA, bf, bcs);
    auto fB = Field(exec, 1, 4.0);
    auto dummy = Dummy(vf);

    NeoFOAM::Dictionary dict;
    NeoFOAM::Dictionary subDict;
    subDict.insert("type", std::string("explicitRungeKutta"));
    subDict.insert("Relative Tolerance", NeoFOAM::scalar(1.e-5));
    subDict.insert("Absolute Tolerance", NeoFOAM::scalar(1.e-10));
    subDict.insert("Fixed Step Size", NeoFOAM::scalar(1.0e-3));
    subDict.insert("End Time", NeoFOAM::scalar(0.005));
    dict.insert("ddtSchemes", subDict);

    // Operator ddtOperator = NeoFOAM::dsl::temporal::Ddt(exec, vf);
    // auto eqn = ddtOperator + dummy;
    // eqn.solve(vf, dict);
}
