// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/fieldCollection.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/oldTimeCollection.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
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

struct CreateField
{
    std::string name;
    NeoFOAM::UnstructuredMesh mesh;
    std::size_t timeIndex;
    std::size_t iterationIndex;
    std::int64_t subCycleIndex;
    NeoFOAM::Document operator()(NeoFOAM::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
        for (auto patchi : std::vector<size_t> {0, 1, 2, 3})
        {
            NeoFOAM::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 2.0);
            bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(mesh, dict, patchi));
        }
        NeoFOAM::Field<NeoFOAM::scalar> internalField(mesh.exec(), mesh.nCells(), 1.0);
        fvcc::VolumeField<NeoFOAM::scalar> vf(mesh.exec(), name, mesh, internalField, bcs, db, "", "");
        return NeoFOAM::Document(
            {{"name", vf.name},
                {"timeIndex", timeIndex},
                {"iterationIndex", iterationIndex},
                {"subCycleIndex", subCycleIndex},
                {"field", vf}},
            fvcc::validateFieldDoc
        );
    }
};

TEST_CASE("FieldCollection")
{
    NeoFOAM::Database db;

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("oldTime")
    {
        fvcc::FieldCollection::create(db, "testFieldCollection");

        fvcc::FieldCollection fieldCollection = fvcc::FieldCollection::get(db, "testFieldCollection");
        fvcc::VolumeField<NeoFOAM::scalar>& T = fieldCollection.registerField<fvcc::VolumeField<NeoFOAM::scalar>>(
            CreateField{
                .name = "T",
                .mesh = mesh,
                .timeIndex = 1,
                .iterationIndex = 1,
                .subCycleIndex = 1
            }
        );

        REQUIRE(T.name == "T");
        REQUIRE(T.hasDatabase());

        SECTION("oldTimeCollection")
        {
            fvcc::OldTimeCollection::create(db, "testOldTimeCollection");
        }
        SECTION("create oldTimeCollection: " + execName)
        {
            fvcc::OldTimeCollection::create(db, "testOldTimeCollection");

            auto& oldTimeCollection = fvcc::OldTimeCollection::getCollection(db, "testOldTimeCollection");
            REQUIRE(oldTimeCollection.size() == 0);
        }
        
        SECTION("usage")
        {
            auto& Told = fvcc::oldTime(T);

            REQUIRE(Told.name == "T_0");
            REQUIRE(Told.internalField().copyToHost()[0] == 1.0);

            auto& sameTold = fvcc::oldTime(T);
            // check if the same field is returned
            REQUIRE(&Told == &sameTold);

            auto& Told2 = fvcc::oldTime(Told);
            REQUIRE(Told2.name == "T_0_0");
            REQUIRE(Told2.internalField().copyToHost()[0] == 1.0);

            auto& sameTold2 = fvcc::oldTime(Told);
            // check if the same field is returned
            REQUIRE(&Told2 == &sameTold2);
        }
        
    }
}
