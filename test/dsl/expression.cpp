// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "common.hpp"
#include "NeoFOAM/NeoFOAM.hpp"

using Expression = NeoFOAM::dsl::Expression<NeoFOAM::scalar>;

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


    SECTION("Create equation and perform explicitOperation on " + execName)
    {
        // TODO conversion from Dummy to SpatialOperator is not automatic
        dsl::SpatialOperator<NeoFOAM::scalar> a = Dummy<NeoFOAM::scalar>(vf);
        dsl::SpatialOperator<NeoFOAM::scalar> b = Dummy<NeoFOAM::scalar>(vf);

        auto eqnA = a + b;
        auto eqnB = fB * dsl::SpatialOperator<NeoFOAM::scalar>(Dummy<NeoFOAM::scalar>(vf))
                  + 2 * dsl::SpatialOperator<NeoFOAM::scalar>(Dummy<NeoFOAM::scalar>(vf));
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

    SECTION("Create equation and perform implicitOperation on " + execName)
    {
        // TODO conversion from Dummy to SpatialOperator is not automatic
        dsl::SpatialOperator<NeoFOAM::scalar> a =
            Dummy<NeoFOAM::scalar>(vf, Operator::Type::Implicit);
        dsl::SpatialOperator<NeoFOAM::scalar> b =
            Dummy<NeoFOAM::scalar>(vf, Operator::Type::Implicit);

        auto eqnA = a + b;
        auto eqnB = fB
                      * dsl::SpatialOperator<NeoFOAM::scalar>(
                          Dummy<NeoFOAM::scalar>(vf, Operator::Type::Implicit)
                      )
                  + 2
                        * dsl::SpatialOperator<NeoFOAM::scalar>(
                            Dummy<NeoFOAM::scalar>(vf, Operator::Type::Implicit)
                        );
        auto eqnC = Expression(2 * a - b);
        auto eqnD = 3 * (2 * a - b);
        auto eqnE = (2 * a - b) + (2 * a - b);
        auto eqnF = (2 * a - b) - (2 * a - b);

        REQUIRE(eqnA.size() == 2);
        REQUIRE(eqnB.size() == 2);
        REQUIRE(eqnC.size() == 2);

        REQUIRE(getDiag(eqnA.implicitOperation()) == 4);
        REQUIRE(getRhs(eqnA.implicitOperation()) == 4);

        REQUIRE(getDiag(eqnB.implicitOperation()) == 12);
        REQUIRE(getRhs(eqnB.implicitOperation()) == 12);

        REQUIRE(getDiag(eqnC.implicitOperation()) == 2);
        REQUIRE(getRhs(eqnC.implicitOperation()) == 2);

        REQUIRE(getDiag(eqnD.implicitOperation()) == 6);
        REQUIRE(getRhs(eqnD.implicitOperation()) == 6);

        REQUIRE(getDiag(eqnE.implicitOperation()) == 4);
        REQUIRE(getRhs(eqnE.implicitOperation()) == 4);

        REQUIRE(getDiag(eqnF.implicitOperation()) == 0);
        REQUIRE(getRhs(eqnF.implicitOperation()) == 0);
    }
}
