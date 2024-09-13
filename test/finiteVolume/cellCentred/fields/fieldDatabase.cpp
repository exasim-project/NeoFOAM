// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fieldDatabase.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/solutionFields.hpp"

template<typename T>
using I = std::initializer_list<T>;

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

fvcc::VolumeField<NeoFOAM::scalar> createVolumeField(const NeoFOAM::UnstructuredMesh& mesh)
{
    std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
    for (auto patchi : I<size_t> {0, 1, 2, 3})
    {
        NeoFOAM::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", 2.0);
        bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(mesh, dict, patchi));
    }
    fvcc::VolumeField<NeoFOAM::scalar> vf(mesh.exec(), "vf", mesh, bcs);
    NeoFOAM::fill(vf.internalField(), 1.0);
    return vf;
}


TEST_CASE("FieldDatabase")
{

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("store Field: " + execName)
    {
        auto vf = createVolumeField(mesh);
        NeoFOAM::fill(vf.internalField(), 1.0);

        fvcc::FieldDatabase fieldDB {};

        fieldDB.insert("test", 1.0);
        REQUIRE(fieldDB.get<double>("test") == 1.0);

        fieldDB.insert("vf", vf);
        auto vf2 = fieldDB.get<fvcc::VolumeField<NeoFOAM::scalar>>("vf").internalField().copyToHost();
        REQUIRE(vf2.span()[0] == 1.0);
    }

}


TEST_CASE("SolutionFields")
{

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("SolutionFields: " + execName)
    {
        fvcc::SolutionFields fm = fvcc::operations::newFieldEntity(createVolumeField(mesh));

        auto fieldPtr = fm.get<fvcc::FieldComponent<fvcc::VolumeField<NeoFOAM::scalar>>>(0).field;//->internalField().copyToHost();
        auto hostInteralField = fieldPtr->internalField().copyToHost();
        REQUIRE(hostInteralField.span()[0] == 1.0);
        REQUIRE(fm.get<fvcc::FieldComponent<fvcc::VolumeField<NeoFOAM::scalar>>>(0).timeIndex == 0.0);
        REQUIRE(fm.get<fvcc::FieldComponent<fvcc::VolumeField<NeoFOAM::scalar>>>(0).iterationIndex == 0.0);
        REQUIRE(fm.get<fvcc::FieldComponent<fvcc::VolumeField<NeoFOAM::scalar>>>(0).category == fvcc::FieldCategory::newTime);
    }
}

