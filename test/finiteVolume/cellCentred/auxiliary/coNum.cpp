// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/boundary.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/auxiliary/coNum.hpp"

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("Courant Number")
{
    namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("can determine maximum courant number of flux field on 1D uniform mesh: " + execName)
    {
        NeoFOAM::UnstructuredMesh mesh = NeoFOAM::create1DUniformMesh(exec, 4);
        std::vector<fvcc::SurfaceBoundary<NeoFOAM::scalar>> bcs {};
        for (auto patchi : I<NeoFOAM::size_t> {0, 1})
        {
            NeoFOAM::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 1.0);
            bcs.push_back(fvcc::SurfaceBoundary<NeoFOAM::scalar>(mesh, dict, patchi));
        }

        fvcc::SurfaceField<NeoFOAM::scalar> sf(exec, "sf", mesh, bcs);
        NeoFOAM::fill(sf.internalField(), 1.0);
        sf.correctBoundaryConditions();

        // use arbitrary time step size of 0.01
        const NeoFOAM::scalar coNum = fvcc::computeCoNum(sf, 0.01);

        REQUIRE(coNum == 0.04);
    }
}
