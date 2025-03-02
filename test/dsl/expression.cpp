// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include "common.hpp"

namespace dsl = NeoFOAM::dsl;


// TEST_CASE("Expression")
TEMPLATE_TEST_CASE("Expression", "[template]", NeoFOAM::scalar, NeoFOAM::Vector)
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    auto mesh = NeoFOAM::createSingleCellMesh(exec);

    const size_t size {1};
    NeoFOAM::BoundaryFields<TestType> bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

    std::vector<fvcc::VolumeBoundary<TestType>> bcs {};
    NeoFOAM::Field<TestType> fA(exec, 1, 2.0 * NeoFOAM::one<TestType>::value);
    NeoFOAM::Field<NeoFOAM::scalar> scaleField(exec, 1, 4.0);
    auto vf = fvcc::VolumeField(exec, "vf", mesh, fA, bf, bcs);


    SECTION("Create equation and perform explicitOperation on " + execName)
    {
        // TODO conversion from Dummy to SpatialOperator is not automatic
        dsl::SpatialOperator<TestType> a = Dummy<TestType>(vf);
        dsl::SpatialOperator<TestType> b = Dummy<TestType>(vf);

        auto eqnA = a + b;
        auto eqnB = scaleField * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf))
                  + 2 * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf));
        auto eqnC = dsl::Expression<TestType>(2 * a - b);
        auto eqnD = 3 * (2 * a - b);
        auto eqnE = (2 * a - b) + (2 * a - b);
        auto eqnF = (2 * a - b) - (2 * a - b);

        REQUIRE(eqnA.size() == 2);
        REQUIRE(eqnB.size() == 2);
        REQUIRE(eqnC.size() == 2);

        // 2 + 2 = 4
        REQUIRE(getField(eqnA.explicitOperation(size)) == 4 * NeoFOAM::one<TestType>::value);
        // 4*2 + 2*2 = 12
        REQUIRE(getField(eqnB.explicitOperation(size)) == 12 * NeoFOAM::one<TestType>::value);
        // 2*2 - 2 = 2
        REQUIRE(getField(eqnC.explicitOperation(size)) == 2 * NeoFOAM::one<TestType>::value);
        // 3*(2*2 - 2) = 6
        REQUIRE(getField(eqnD.explicitOperation(size)) == 6 * NeoFOAM::one<TestType>::value);
        // 2*2 - 2 + 2*2 - 2 = 4
        REQUIRE(getField(eqnE.explicitOperation(size)) == 4 * NeoFOAM::one<TestType>::value);
        // 2*2 - 2 - 2*2 + 2 = 0
        REQUIRE(getField(eqnF.explicitOperation(size)) == 0 * NeoFOAM::one<TestType>::value);
    }

    SECTION("Create equation and perform implicitOperation on " + execName)
    {
        // TODO conversion from Dummy to SpatialOperator is not automatic
        dsl::SpatialOperator<TestType> a = Dummy<TestType>(vf, Operator::Type::Implicit);
        dsl::SpatialOperator<TestType> b = Dummy<TestType>(vf, Operator::Type::Implicit);

        auto eqnA = a + b;
        auto eqnB =
            scaleField
                * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf, Operator::Type::Implicit))
            + 2 * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf, Operator::Type::Implicit));
        auto eqnC = dsl::Expression<TestType>(2 * a - b);
        auto eqnD = 3 * (2 * a - b);
        auto eqnE = (2 * a - b) + (2 * a - b);
        auto eqnF = (2 * a - b) - (2 * a - b);

        REQUIRE(eqnA.size() == 2);
        REQUIRE(eqnB.size() == 2);
        REQUIRE(eqnC.size() == 2);

        // 2 + 2 = 4
        REQUIRE(getDiag(eqnA.implicitOperation()) == 4 * NeoFOAM::one<TestType>::value);
        REQUIRE(getRhs(eqnA.implicitOperation()) == 4 * NeoFOAM::one<TestType>::value);

        // 4*2 + 2*2 = 12
        REQUIRE(getDiag(eqnB.implicitOperation()) == 12 * NeoFOAM::one<TestType>::value);
        REQUIRE(getRhs(eqnB.implicitOperation()) == 12 * NeoFOAM::one<TestType>::value);

        // 2*2 - 2 = 2
        REQUIRE(getDiag(eqnC.implicitOperation()) == 2 * NeoFOAM::one<TestType>::value);
        REQUIRE(getRhs(eqnC.implicitOperation()) == 2 * NeoFOAM::one<TestType>::value);

        // 3*(2*2 - 2) = 6
        REQUIRE(getDiag(eqnD.implicitOperation()) == 6 * NeoFOAM::one<TestType>::value);
        REQUIRE(getRhs(eqnD.implicitOperation()) == 6 * NeoFOAM::one<TestType>::value);

        // 2*2 - 2 + 2*2 - 2 = 4
        REQUIRE(getDiag(eqnE.implicitOperation()) == 4 * NeoFOAM::one<TestType>::value);
        REQUIRE(getRhs(eqnE.implicitOperation()) == 4 * NeoFOAM::one<TestType>::value);

        // 2*2 - 2 - 2*2 + 2 = 0
        REQUIRE(getDiag(eqnF.implicitOperation()) == 0 * NeoFOAM::one<TestType>::value);
        REQUIRE(getRhs(eqnF.implicitOperation()) == 0 * NeoFOAM::one<TestType>::value);
    }
}
