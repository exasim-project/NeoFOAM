// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_approx.hpp>


#include "NeoFOAM/NeoFOAM.hpp"

using NeoFOAM::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoFOAM::finiteVolume::cellCentred::VolumeField;
using NeoFOAM::finiteVolume::cellCentred::SurfaceField;

namespace NeoFOAM
{

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("laplacianOperator fixedValue")
{
    const size_t nCells = 10;
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    auto mesh = create1DUniformMesh(exec, nCells);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoFOAM::scalar>>(mesh);

    fvcc::SurfaceField<NeoFOAM::scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalField(), 2.0);

    SECTION("fixedValue")
    {
        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs;
        bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(
            mesh,
            NeoFOAM::Dictionary(
                {{"type", std::string("fixedValue")}, {"fixedValue", NeoFOAM::scalar(0.5)}}
            ),
            0
        ));
        bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(
            mesh,
            NeoFOAM::Dictionary(
                {{"type", std::string("fixedValue")}, {"fixedValue", NeoFOAM::scalar(10.5)}}
            ),
            1
        ));

        fvcc::VolumeField<NeoFOAM::scalar> phi(exec, "phi", mesh, bcs);
        NeoFOAM::parallelFor(
            phi.internalField(), KOKKOS_LAMBDA(const size_t i) { return scalar(i + 1); }
        );
        phi.correctBoundaryConditions();

        SECTION("Construct from Token" + execName)
        {
            NeoFOAM::Input input = NeoFOAM::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            );
            fvcc::LaplacianOperator(dsl::Operator::Type::Implicit, gamma, phi, input);
        }

        SECTION("explicit laplacian operator" + execName)
        {
            NeoFOAM::Input input = NeoFOAM::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            );
            fvcc::LaplacianOperator lapOp(dsl::Operator::Type::Explicit, gamma, phi, input);
            Field<NeoFOAM::scalar> source(exec, nCells, 0.0);
            lapOp.explicitOperation(source);
            auto sourceHost = source.copyToHost();
            auto sSource = sourceHost.span();
            for (size_t i = 0; i < nCells; i++)
            {
                // the laplacian of a linear function is 0
                REQUIRE(sourceHost[i] == Catch::Approx(0.0).margin(1e-8));
            }
        }

        SECTION("implicit laplacian operator" + execName)
        {
            NeoFOAM::Input input = NeoFOAM::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            );
            fvcc::LaplacianOperator lapOp(dsl::Operator::Type::Implicit, gamma, phi, input);
            auto ls = lapOp.createEmptyLinearSystem();
            lapOp.implicitOperation(ls);
            fvcc::LinearSystem<NeoFOAM::scalar> ls2(
                phi, ls, fvcc::SparsityPattern::readOrCreate(mesh)
            );


            auto result = ls2 & phi;
            auto resultHost = result.internalField().copyToHost();
            auto sResult = resultHost.span();
            for (size_t celli = 0; celli < sResult.size(); celli++)
            {
                // the laplacian of a linear function is 0
                REQUIRE(sResult[celli] == Catch::Approx(0.0).margin(1e-8));
            }
        }

        SECTION("implicit laplacian operator scale" + execName)
        {
            NeoFOAM::Input input = NeoFOAM::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            );
            dsl::SpatialOperator lapOp = dsl::imp::laplacian(gamma, phi);
            lapOp.build(input);
            lapOp = dsl::Coeff(-0.5) * lapOp;
            auto ls = lapOp.createEmptyLinearSystem();
            lapOp.implicitOperation(ls);
            fvcc::LinearSystem<NeoFOAM::scalar> ls2(
                phi, ls, fvcc::SparsityPattern::readOrCreate(mesh)
            );


            auto result = ls2 & phi;
            auto resultHost = result.internalField().copyToHost();
            auto sResult = resultHost.span();
            for (size_t celli = 0; celli < sResult.size(); celli++)
            {
                // the laplacian of a linear function is 0
                REQUIRE(sResult[celli] == Catch::Approx(0.0).margin(1e-8));
            }
        }
    }
}

TEST_CASE("laplacianOperator fixedGradient")
{
    const size_t nCells = 10;
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    auto mesh = create1DUniformMesh(exec, nCells);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoFOAM::scalar>>(mesh);

    fvcc::SurfaceField<NeoFOAM::scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalField(), 2.0);

    SECTION("fixedGradient")
    {
        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs;
        bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(
            mesh,
            NeoFOAM::Dictionary(
                {{"type", std::string("fixedGradient")}, {"fixedGradient", NeoFOAM::scalar(-10.0)}}
            ),
            0
        ));
        bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(
            mesh,
            NeoFOAM::Dictionary(
                {{"type", std::string("fixedGradient")}, {"fixedGradient", NeoFOAM::scalar(10.0)}}
            ),
            1
        ));

        fvcc::VolumeField<NeoFOAM::scalar> phi(exec, "phi", mesh, bcs);
        NeoFOAM::parallelFor(
            phi.internalField(), KOKKOS_LAMBDA(const size_t i) { return scalar(i + 1); }
        );
        phi.correctBoundaryConditions();

        SECTION("Construct from Token" + execName)
        {
            NeoFOAM::Input input = NeoFOAM::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            );
            fvcc::LaplacianOperator(dsl::Operator::Type::Implicit, gamma, phi, input);
        }

        SECTION("explicit laplacian operator" + execName)
        {
            NeoFOAM::Input input = NeoFOAM::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            );
            fvcc::LaplacianOperator lapOp(dsl::Operator::Type::Explicit, gamma, phi, input);
            Field<NeoFOAM::scalar> source(exec, nCells, 0.0);
            lapOp.explicitOperation(source);
            auto sourceHost = source.copyToHost();
            auto sSource = sourceHost.span();
            for (size_t i = 0; i < nCells; i++)
            {
                // the laplacian of a linear function is 0
                REQUIRE(sourceHost[i] == Catch::Approx(0.0).margin(1e-8));
            }
        }

        SECTION("implicit laplacian operator" + execName)
        {
            NeoFOAM::Input input = NeoFOAM::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            );
            fvcc::LaplacianOperator lapOp(dsl::Operator::Type::Explicit, gamma, phi, input);
            auto ls = lapOp.createEmptyLinearSystem();
            lapOp.implicitOperation(ls);
            fvcc::LinearSystem<NeoFOAM::scalar> ls2(
                phi, ls, fvcc::SparsityPattern::readOrCreate(mesh)
            );


            auto result = ls2 & phi;
            auto resultHost = result.internalField().copyToHost();
            auto sResult = resultHost.span();
            for (size_t celli = 0; celli < sResult.size(); celli++)
            {
                // the laplacian of a linear function is 0
                REQUIRE(sResult[celli] == Catch::Approx(0.0).margin(1e-8));
            }
        }
    }
}


} // namespace NeoFOAM
