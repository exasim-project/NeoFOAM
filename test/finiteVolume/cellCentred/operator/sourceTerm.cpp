// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

using Operator = NeoFOAM::dsl::Operator;

namespace NeoFOAM
{

TEMPLATE_TEST_CASE("SourceTerm", "[template]", NeoFOAM::scalar, NeoFOAM::Vector)
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    auto mesh = createSingleCellMesh(exec);

    auto coeffBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> coeff(exec, "coeff", mesh, coeffBCs);
    fill(coeff.internalField(), 2.0);
    fill(coeff.boundaryField().value(), 0.0);
    coeff.correctBoundaryConditions();

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    fill(phi.internalField(), 10 * one<TestType>());
    fill(phi.boundaryField().value(), zero<TestType>());
    phi.correctBoundaryConditions();


    SECTION("explicit SourceTerm" + execName)
    {
        fvcc::SourceTerm<TestType> sTerm(Operator::Type::Explicit, coeff, phi);

        auto source = Field<TestType>(exec, phi.size(), zero<TestType>());
        sTerm.explicitOperation(source);

        // cell has one cell
        auto hostSource = source.copyToHost();
        for (auto ii = 0; ii < hostSource.size(); ++ii)
        {
            REQUIRE(hostSource[ii] - 20 * one<TestType>() == TestType(0.0));
        }
    }

    SECTION("implicit SourceTerm" + execName)
    {
        fvcc::SourceTerm<TestType> sTerm(Operator::Type::Implicit, coeff, phi);
        auto ls = sTerm.createEmptyLinearSystem();
        sTerm.implicitOperation(ls);
        auto lsHost = ls.copyToHost();
        auto vol = mesh.cellVolumes().copyToHost();
        const auto& values = lsHost.matrix().values();

        for (auto ii = 0; ii < values.size(); ++ii)
        {
            REQUIRE(values[ii] - 2 * vol[0] * one<TestType>() == TestType(0.0));
        }
    }
}

}
