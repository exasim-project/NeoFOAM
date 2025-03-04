// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include "common.hpp"


namespace dsl = NeoFOAM::dsl;


TEMPLATE_TEST_CASE("SpatialOperator", "[template]", NeoFOAM::scalar, NeoFOAM::Vector)
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    auto mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("SpatialOperator creation on " + execName)
    {
        NeoFOAM::Field<TestType> fA(exec, 1, 2.0 * NeoFOAM::one<TestType>());
        NeoFOAM::BoundaryFields<TestType> bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

        std::vector<fvcc::VolumeBoundary<TestType>> bcs {};
        auto vf = fvcc::VolumeField<TestType>(exec, "vf", mesh, fA, bf, bcs);
        dsl::SpatialOperator<TestType> b = Dummy<TestType>(vf);

        REQUIRE(b.getName() == "Dummy");
        REQUIRE(b.getType() == Operator::Type::Explicit);
    }

    SECTION("Supports Coefficients Explicit " + execName)
    {
        std::vector<fvcc::VolumeBoundary<TestType>> bcs {};

        NeoFOAM::Field<TestType> fA(exec, 1, 2.0 * NeoFOAM::one<TestType>());
        NeoFOAM::Field<NeoFOAM::scalar> scaleField(exec, 1, 2.0);
        NeoFOAM::BoundaryFields<TestType> bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());
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
        NeoFOAM::Field<TestType> fA(exec, 1, 2.0 * NeoFOAM::one<TestType>());
        NeoFOAM::BoundaryFields<TestType> bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

        std::vector<fvcc::VolumeBoundary<TestType>> bcs {};
        auto vf = fvcc::VolumeField(exec, "vf", mesh, fA, bf, bcs);
        dsl::SpatialOperator<TestType> b = Dummy<TestType>(vf, Operator::Type::Implicit);

        REQUIRE(b.getName() == "Dummy");
        REQUIRE(b.getType() == Operator::Type::Implicit);

        auto ls = b.createEmptyLinearSystem();
        REQUIRE(ls.matrix().nValues() == 1);
        REQUIRE(ls.matrix().nColIdxs() == 1);
        REQUIRE(ls.matrix().nRows() == 1);
    }

    SECTION("Supports Coefficients Implicit " + execName)
    {
        std::vector<fvcc::VolumeBoundary<TestType>> bcs {};

        NeoFOAM::Field<TestType> fA(exec, 1, 2.0 * NeoFOAM::one<TestType>());
        NeoFOAM::Field<NeoFOAM::scalar> scaleField(exec, 1, 2.0);
        NeoFOAM::BoundaryFields<TestType> bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());
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

        // Field source(exec, 1, 2.0);
        auto ls = c.createEmptyLinearSystem();
        c.implicitOperation(ls);

        // c = 2 * 2
        auto hostRhsC = ls.rhs().copyToHost();
        REQUIRE(hostRhsC.span()[0] == 4.0 * NeoFOAM::one<TestType>());
        auto hostLsC = ls.copyToHost();
        REQUIRE(hostLsC.matrix().values()[0] == 4.0 * NeoFOAM::one<TestType>());


        // d= 2 * 2
        ls = d.createEmptyLinearSystem();
        d.implicitOperation(ls);
        auto hostRhsD = ls.rhs().copyToHost();
        REQUIRE(hostRhsD.span()[0] == 4.0 * NeoFOAM::one<TestType>());
        auto hostLsD = ls.copyToHost();
        REQUIRE(hostLsD.matrix().values()[0] == 4.0 * NeoFOAM::one<TestType>());


        // e = - -3 * 2 * 2 = -12
        ls = e.createEmptyLinearSystem();
        e.implicitOperation(ls);
        auto hostRhsE = ls.rhs().copyToHost();
        REQUIRE(hostRhsE.span()[0] == -12.0 * NeoFOAM::one<TestType>());
        auto hostLsE = ls.copyToHost();
        REQUIRE(hostLsE.matrix().values()[0] == -12.0 * NeoFOAM::one<TestType>());
    }
}
