// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

using NeoFOAM::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoFOAM::finiteVolume::cellCentred::VolumeField;
using NeoFOAM::finiteVolume::cellCentred::SurfaceField;
using NeoFOAM::Input;

template<typename T>
using I = std::initializer_list<T>;

TEMPLATE_TEST_CASE("upwind", "", NeoFOAM::scalar, NeoFOAM::Vector)
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    auto mesh = NeoFOAM::create1DUniformMesh(exec, 10);
    Input input = NeoFOAM::TokenList({std::string("upwind")});
    auto upwind = SurfaceInterpolation(exec, mesh, input);
    std::vector<fvcc::SurfaceBoundary<TestType>> bcs {};
    for (auto patchi : I<size_t> {0, 1})
    {
        NeoFOAM::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", NeoFOAM::one<TestType>::value);
        bcs.push_back(fvcc::SurfaceBoundary<TestType>(mesh, dict, patchi));
    }

    auto in = VolumeField<TestType>(exec, "in", mesh, {});
    auto flux = SurfaceField<NeoFOAM::scalar>(exec, "flux", mesh, {});
    auto out = SurfaceField<TestType>(exec, "out", mesh, bcs);

    fill(flux.internalField(), NeoFOAM::one<NeoFOAM::scalar>::value);
    fill(in.internalField(), NeoFOAM::one<TestType>::value);

    upwind.interpolate(flux, in, out);
    out.correctBoundaryConditions();

    for (int i = 0; i < out.internalField().size(); i++)
    {
        REQUIRE(out.internalField()[i] == NeoFOAM::one<TestType>::value);
    }
}
