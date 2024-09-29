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

// #include "customTestDocument.hpp"

#include "NeoFOAM/core/database.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

fvcc::VolumeField<NeoFOAM::scalar>
createVolumeField(const NeoFOAM::UnstructuredMesh& mesh, std::string fieldName)
{
    std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
    for (auto patchi : std::vector<size_t> {0, 1, 2, 3})
    {
        NeoFOAM::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", 2.0);
        bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(mesh, dict, patchi));
    }
    fvcc::VolumeField<NeoFOAM::scalar> vf(mesh.exec(), fieldName, mesh, bcs);
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

    SECTION("create FieldDatabase: " + execName)
    {

        fvcc::FieldDatabase fieldDB {};

        SECTION("create SolutionField")
        {
            auto solutionField =
                fieldDB.createSolutionField([&]() { return createVolumeField(mesh, "T"); });
            REQUIRE(solutionField.name() == "T");
            REQUIRE(solutionField.field().internalField().copyToHost()[0] == 1.0);

            fvcc::VolumeField<NeoFOAM::scalar>& volField =
                fieldDB.createSolutionField([&]() { return createVolumeField(mesh, "T2"); }
                ).field();

            REQUIRE(volField.name == "T2");
            REQUIRE(volField.internalField().copyToHost()[0] == 1.0);
        }

        SECTION("create oldTime")
        {
            auto& T =
                fieldDB.createSolutionField([&]() { return createVolumeField(mesh, "T"); }).field();
            REQUIRE(T.name == "T");
            REQUIRE(T.hasSolField());

            auto& Told = fvcc::operations::oldTime(T);
            REQUIRE(Told.name == "T_0");
            REQUIRE(Told.internalField().copyToHost()[0] == 1.0);

            auto& sameTold = fvcc::operations::oldTime(T);
            // check if the same field is returned
            REQUIRE(&Told == &sameTold);

            auto& Told2 = fvcc::operations::oldTime(Told);
            REQUIRE(Told2.name == "T_0_0");
            REQUIRE(Told2.internalField().copyToHost()[0] == 1.0);

            auto& sameTold2 = fvcc::operations::oldTime(Told);
            // check if the same field is returned
            REQUIRE(&Told2 == &sameTold2);
        }
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
        // fvcc::SolutionFields fm = fvcc::operations::newFieldEntity(createVolumeField(mesh));

        // auto fieldPtr =
        // fm.get<fvcc::FieldComponent<fvcc::VolumeField<NeoFOAM::scalar>>>(0).field;//->internalField().copyToHost();
        // auto hostInteralField = fieldPtr->internalField().copyToHost();
        // REQUIRE(hostInteralField.span()[0] == 1.0);
        // REQUIRE(fm.get<fvcc::FieldComponent<fvcc::VolumeField<NeoFOAM::scalar>>>(0).timeIndex ==
        // 0.0);
        // REQUIRE(fm.get<fvcc::FieldComponent<fvcc::VolumeField<NeoFOAM::scalar>>>(0).iterationIndex
        // == 0.0);
        // REQUIRE(fm.get<fvcc::FieldComponent<fvcc::VolumeField<NeoFOAM::scalar>>>(0).category ==
        // fvcc::FieldCategory::newTime);
    }
}
