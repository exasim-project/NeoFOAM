// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "common.hpp"

namespace dsl = NeoFOAM::dsl;


TEMPLATE_TEST_CASE("SpatialOperator", "[template]", NeoFOAM::scalar, NeoFOAM::Vector)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = NeoFOAM::createSingleCellMesh(exec);
    auto sp = NeoFOAM::finiteVolume::cellCentred::SparsityPattern {mesh};

    auto fA = NeoFOAM::Field<TestType>(exec, 1, 2.0 * NeoFOAM::one<TestType>());
    auto bf = NeoFOAM::BoundaryFields<TestType>(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());
    auto bcs = std::vector<fvcc::VolumeBoundary<TestType>> {};
    auto scaleField = NeoFOAM::Field<NeoFOAM::scalar>(exec, 1, 2.0);

    SECTION("SpatialOperator creation on " + execName)
    {
        auto vf = fvcc::VolumeField<TestType>(exec, "vf", mesh, fA, bf, bcs);
        dsl::SpatialOperator<TestType> b = Dummy<TestType>(vf);

        REQUIRE(b.getName() == "Dummy");
        REQUIRE(b.getType() == Operator::Type::Explicit);
    }

    SECTION("Supports Coefficients Explicit " + execName)
    {
        auto vf = fvcc::VolumeField<TestType>(exec, "vf", mesh, fA, bf, bcs);

        dsl::SpatialOperator c = 2.0 * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf));
        dsl::SpatialOperator d = scaleField * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf));
        dsl::SpatialOperator e =
            Coeff(-3, scaleField) * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf));

        [[maybe_unused]] auto coeffC = c.getCoefficient();
        [[maybe_unused]] auto coeffD = d.getCoefficient();
        [[maybe_unused]] auto coeffE = e.getCoefficient();

        NeoFOAM::Field<TestType> source(exec, 1, 2.0 * NeoFOAM::one<TestType>());
        c.explicitOperation(source);

        // 2 += 2 * 2
        auto hostSourceC = source.copyToHost();
        REQUIRE(hostSourceC.span()[0] == 6.0 * NeoFOAM::one<TestType>());

        // 6 += 2 * 2
        d.explicitOperation(source);
        auto hostSourceD = source.copyToHost();
        REQUIRE(hostSourceD.span()[0] == 10.0 * NeoFOAM::one<TestType>());

        // 10 += - 6 * 2
        e.explicitOperation(source);
        auto hostSourceE = source.copyToHost();
        REQUIRE(hostSourceE.span()[0] == -2.0 * NeoFOAM::one<TestType>());
    }

    SECTION("Implicit Operations " + execName)
    {
        auto vf = fvcc::VolumeField(exec, "vf", mesh, fA, bf, bcs);
        dsl::SpatialOperator<TestType> b = Dummy<TestType>(vf, Operator::Type::Implicit);

        REQUIRE(b.getName() == "Dummy");
        REQUIRE(b.getType() == Operator::Type::Implicit);

        // auto ls = b.createEmptyLinearSystem();
        // REQUIRE(ls.matrix().nNonZeros() == 1);
        // REQUIRE(ls.matrix().nRows() == 1);
    }

    SECTION("Supports Coefficients Implicit " + execName)
    {
        auto vf = fvcc::VolumeField(exec, "vf", mesh, fA, bf, bcs);

        dsl::SpatialOperator c =
            2 * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf, Operator::Type::Implicit));
        dsl::SpatialOperator d =
            scaleField
            * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf, Operator::Type::Implicit));
        dsl::SpatialOperator e =
            Coeff(-3, scaleField)
            * dsl::SpatialOperator<TestType>(Dummy<TestType>(vf, Operator::Type::Implicit));

        [[maybe_unused]] auto coeffC = c.getCoefficient();
        [[maybe_unused]] auto coeffD = d.getCoefficient();
        [[maybe_unused]] auto coeffE = e.getCoefficient();

        auto ls = NeoFOAM::la::createEmptyLinearSystem<
            TestType,
            NeoFOAM::localIdx,
            NeoFOAM::finiteVolume::cellCentred::SparsityPattern>(sp);
        Field source(exec, 1, 2.0);
        c.implicitOperation(ls);

        // c = 2 * 2
        auto hostRhsC = ls.rhs().copyToHost();
        auto hostLsC = ls.copyToHost();
        REQUIRE(hostRhsC.span()[0] == 4.0 * NeoFOAM::one<TestType>());
        REQUIRE(hostLsC.matrix().values()[0] == 4.0 * NeoFOAM::one<TestType>());

        // d= 2 * 2
        ls.reset();
        d.implicitOperation(ls);
        auto hostRhsD = ls.rhs().copyToHost();
        auto hostLsD = ls.copyToHost();
        REQUIRE(hostRhsD.span()[0] == 4.0 * NeoFOAM::one<TestType>());
        REQUIRE(hostLsD.matrix().values()[0] == 4.0 * NeoFOAM::one<TestType>());

        // e = - -3 * 2 * 2 = -12
        ls.reset();
        e.implicitOperation(ls);
        auto hostRhsE = ls.rhs().copyToHost();
        auto hostLsE = ls.copyToHost();
        REQUIRE(hostRhsE.span()[0] == -12.0 * NeoFOAM::one<TestType>());
        REQUIRE(hostLsE.matrix().values()[0] == -12.0 * NeoFOAM::one<TestType>());
    }
}
