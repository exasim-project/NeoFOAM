// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "common.hpp"

TEST_CASE("Operator")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    auto mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("Operator creation on " + execName)
    {
        Field fA(exec, 1, 2.0);
        BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
        auto vf = VolumeField(exec, "vf", mesh, fA, bf, bcs);
        auto b = Dummy(vf);

        REQUIRE(b.getName() == "Dummy");
        REQUIRE(b.getType() == Operator::Type::Explicit);
    }

    SECTION("Supports Coefficients Explicit" + execName)
    {
        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};

        Field fA(exec, 1, 2.0);
        Field fB(exec, 1, 2.0);
        BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());
        auto vf = VolumeField(exec, "vf", mesh, fA, bf, bcs);

        auto c = 2 * Dummy(vf);
        auto d = fB * Dummy(vf);
        auto e = Coeff(-3, fB) * Dummy(vf);

        [[maybe_unused]] auto coeffC = c.getCoefficient();
        [[maybe_unused]] auto coeffD = d.getCoefficient();
        [[maybe_unused]] auto coeffE = e.getCoefficient();

        Field source(exec, 1, 2.0);
        c.explicitOperation(source);

        // 2 += 2 * 2
        auto hostSourceC = source.copyToHost();
        REQUIRE(hostSourceC.span()[0] == 6.0);

        // 6 += 2 * 2
        d.explicitOperation(source);
        auto hostSourceD = source.copyToHost();
        REQUIRE(hostSourceD.span()[0] == 10.0);

        // 10 += - 6 * 2
        e.explicitOperation(source);
        auto hostSourceE = source.copyToHost();
        REQUIRE(hostSourceE.span()[0] == -2.0);
    }

    SECTION("Implicit Operations" + execName)
    {
        Field fA(exec, 1, 2.0);
        BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
        auto vf = VolumeField(exec, "vf", mesh, fA, bf, bcs);
        auto b = Dummy(vf, Operator::Type::Implicit);

        REQUIRE(b.getName() == "Dummy");
        REQUIRE(b.getType() == Operator::Type::Implicit);

        auto ls = b.createEmptyLinearSystem();
        REQUIRE(ls.matrix().nValues() == 1);
        REQUIRE(ls.matrix().nColIdxs() == 1);
        REQUIRE(ls.matrix().nRows() == 1);
    }

    SECTION("Supports Coefficients Implicit" + execName)
    {
        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};

        Field fA(exec, 1, 2.0);
        Field fB(exec, 1, 2.0);
        BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());
        auto vf = VolumeField(exec, "vf", mesh, fA, bf, bcs);

        auto c = 2 * Dummy(vf, Operator::Type::Implicit);
        auto d = fB * Dummy(vf, Operator::Type::Implicit);
        auto e = Coeff(-3, fB) * Dummy(vf, Operator::Type::Implicit);

        [[maybe_unused]] auto coeffC = c.getCoefficient();
        [[maybe_unused]] auto coeffD = d.getCoefficient();
        [[maybe_unused]] auto coeffE = e.getCoefficient();

        // Field source(exec, 1, 2.0);
        auto ls = c.createEmptyLinearSystem();
        c.implicitOperation(ls);

        // c = 2 * 2
        auto hostRhsC = ls.rhs().copyToHost();
        REQUIRE(hostRhsC.span()[0] == 4.0);
        auto hostLsC = ls.copyToHost();
        REQUIRE(hostLsC.matrix().values()[0] == 4.0);


        // // d= 2 * 2
        ls = d.createEmptyLinearSystem();
        d.implicitOperation(ls);
        auto hostRhsD = ls.rhs().copyToHost();
        REQUIRE(hostRhsD.span()[0] == 4.0);
        auto hostLsD = ls.copyToHost();
        REQUIRE(hostLsD.matrix().values()[0] == 4.0);


        // e = - -3 * 2 * 2 = -12
        ls = e.createEmptyLinearSystem();
        e.implicitOperation(ls);
        auto hostRhsE = ls.rhs().copyToHost();
        REQUIRE(hostRhsE.span()[0] == -12.0);
        auto hostLsE = ls.copyToHost();
        REQUIRE(hostLsE.matrix().values()[0] == -12.0);
    }
}
