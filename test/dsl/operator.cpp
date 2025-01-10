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

    SECTION("Supports Coefficients" + execName)
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
}
