// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/fields/fieldDatabase.hpp"

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("FieldRepo")
{
    namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("store Field: " + execName)
    {
        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
        for (auto patchi : I<size_t> {0, 1, 2, 3})
        {
            NeoFOAM::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 2.0);
            bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(mesh, dict, patchi));
        }
        fvcc::VolumeField<NeoFOAM::scalar> vf(exec, "vf", mesh, bcs);
        NeoFOAM::fill(vf.internalField(), 1.0);

        NeoFOAM::fieldDataBase fieldDB {};

        fieldDB.insert("test", 1.0);
        REQUIRE(fieldDB.get<double>("test") == 1.0);

        fieldDB.insert("vf", vf);
        auto vf2 = fieldDB.get<fvcc::VolumeField<NeoFOAM::scalar>>("vf").internalField().copyToHost();
        REQUIRE(vf2.span()[0] == 1.0);
    }
}
