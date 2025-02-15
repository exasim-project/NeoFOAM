// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

using NeoFOAM::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoFOAM::finiteVolume::cellCentred::VolumeField;
using NeoFOAM::finiteVolume::cellCentred::SurfaceField;

namespace NeoFOAM
{

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("laplacianOperator")
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

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoFOAM::scalar>>(mesh);
    fvcc::VolumeField<NeoFOAM::scalar> phi(exec, "phi", mesh, volumeBCs);
    NeoFOAM::parallelFor(
        phi.internalField(), KOKKOS_LAMBDA(const size_t i) { return scalar(i + 1); }
    );
    phi.boundaryField().value() = NeoFOAM::Field<NeoFOAM::scalar>(exec, {0.5, 9.5});

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
        for (size_t i = 0; i < nCells; i++)
        {
            REQUIRE(sourceHost[i] == 2.0);
        }
    }
    // auto linear = SurfaceInterpolation(exec, mesh, input);
    // std::vector<fvcc::SurfaceBoundary<TestType>> bcs {};
    // for (auto patchi : I<size_t> {0, 1})
    // {
    //     Dictionary dict;
    //     dict.insert("type", std::string("fixedValue"));
    //     dict.insert("fixedValue", one<TestType>::value);
    //     bcs.push_back(fvcc::SurfaceBoundary<TestType>(mesh, dict, patchi));
    // }

    // auto in = VolumeField<TestType>(exec, "in", mesh, {});
    // auto out = SurfaceField<TestType>(exec, "out", mesh, bcs);

    // fill(in.internalField(), one<TestType>::value);

    // linear.interpolate(in, out);
    // out.correctBoundaryConditions();

    // auto outHost = out.internalField().copyToHost();
    // for (int i = 0; i < out.internalField().size(); i++)
    // {
    //     REQUIRE(outHost[i] == one<TestType>::value);
    // }
}
}
