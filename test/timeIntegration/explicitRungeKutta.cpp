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
#include "NeoFOAM/dsl/solver.hpp"
#include "NeoFOAM/timeIntegration/explicitRungeKutta.hpp"

using namespace NeoFOAM::dsl::temporal;
using namespace NeoFOAM::dsl;


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

    NeoFOAM::Dictionary fvSchemes;
    NeoFOAM::Dictionary ddtSchemes;
    ddtSchemes.insert("type", std::string("explicitRungeKutta"));
    ddtSchemes.insert("Relative Tolerance", NeoFOAM::scalar(1.e-5));
    ddtSchemes.insert("Absolute Tolerance", NeoFOAM::scalar(1.e-10));
    ddtSchemes.insert("Fixed Step Size", NeoFOAM::scalar(1.0e-3));
    ddtSchemes.insert("End Time", NeoFOAM::scalar(0.005));
    fvSchemes.insert("ddtSchemes", ddtSchemes);
    NeoFOAM::Dictionary fvSolution;


    Operator ddtOp = Ddt(vf);
    auto dumb = Dummy(vf);
    auto eqn = ddtOp + dummy;
    double dt {0.1};                           // here the time integrator will deal with this.
    solve(eqn, vf, dt, fvSchemes, fvSolution); // perform 1 step.
}
