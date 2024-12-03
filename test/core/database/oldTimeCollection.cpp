// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core/database/fieldCollection.hpp"
#include "NeoFOAM/core/database/oldTimeCollection.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/core/database/database.hpp"

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
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

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
        fvcc::VolumeField<NeoFOAM::scalar> vf(
            mesh.exec(), name, mesh, internalField, bcs, db, "", ""
        );
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

TEST_CASE("oldTimeCollection")
{
    NeoFOAM::Database db;

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("OldTimeDocument")
    {
        fvcc::OldTimeDocument oldTimeDoc(
            "next",
            "previous",
            "current",
            1
        );

        REQUIRE(oldTimeDoc.nextTime() == "next");
        REQUIRE(oldTimeDoc.previousTime() == "previous");
        REQUIRE(oldTimeDoc.currentTime() == "current");
        REQUIRE(oldTimeDoc.level() == 1);
        REQUIRE(oldTimeDoc.typeName() == "OldTimeDocument");
    }

    SECTION("OldTimeCollection from db and name")
    {
        fvcc::OldTimeCollection& oldTimeCollection =
            fvcc::OldTimeCollection::instance(db, "testFieldCollection_oldTime", "testFieldCollection");

        REQUIRE(oldTimeCollection.name() == "testFieldCollection_oldTime");
        REQUIRE(oldTimeCollection.type() == "OldTimeDocument");
        REQUIRE(oldTimeCollection.db().contains("testFieldCollection_oldTime"));
        REQUIRE(oldTimeCollection.size() == 0);

        fvcc::OldTimeDocument oldTimeDoc(
            "nextTime",
            "previousTime",
            "currentTime",
            1
        );

        oldTimeCollection.insert(oldTimeDoc);
        std::string oldTimeDocKey = oldTimeDoc.id();

        REQUIRE(oldTimeCollection.size() == 1);
        REQUIRE(oldTimeCollection.contains(oldTimeDocKey));
        REQUIRE(oldTimeCollection.findNextTime("nextTime") == oldTimeDocKey);
        REQUIRE(oldTimeCollection.findPreviousTime("previousTime") == oldTimeDocKey);

    }

    SECTION("OldTimeCollection from FieldCollection")
    {
        fvcc::FieldCollection& fieldCollection =
            fvcc::FieldCollection::instance(db, "testFieldCollection");

        auto& oldTimeCollection = fvcc::OldTimeCollection::instance(fieldCollection);
        REQUIRE(oldTimeCollection.name() == "testFieldCollection_oldTime");
        REQUIRE(oldTimeCollection.type() == "OldTimeDocument");
        REQUIRE(oldTimeCollection.db().contains("testFieldCollection_oldTime"));
        REQUIRE(oldTimeCollection.size() == 0);

        const auto& oldTimeCollectionConst = fvcc::OldTimeCollection::instance(fieldCollection);
        REQUIRE(oldTimeCollectionConst.name() == "testFieldCollection_oldTime");

        
    }

    SECTION("oldTime")
    {
        fvcc::FieldCollection& fieldCollection =
            fvcc::FieldCollection::instance(db, "testFieldCollection");
        fvcc::VolumeField<NeoFOAM::scalar>& T =
            fieldCollection.registerField<fvcc::VolumeField<NeoFOAM::scalar>>(
                CreateField {.name = "T", .mesh = mesh, .timeIndex = 1}
            );
        // find the field by name and check the document key
        auto res =
            fieldCollection.find([](const NeoFOAM::Document& doc) { return name(doc) == "T"; });

        REQUIRE(res.size() == 1);
        REQUIRE(T.key == res[0]);
        REQUIRE(T.fieldCollectionName == "testFieldCollection");

        REQUIRE(T.name == "T");
        REQUIRE(T.hasDatabase());
        REQUIRE(T.internalField().copyToHost()[0] == 1.0);
        REQUIRE(T.registered());
        fvcc::FieldDocument& docT = fieldCollection.fieldDoc(T.key);
        // without the timeIndex smaller than 2 the timeIndex overflows if we subtract 1
        // Note: we  are also checking oldTime oldTime(T) in the next section
        REQUIRE(docT.timeIndex() == 1);
        REQUIRE(docT.iterationIndex() == 0);
        REQUIRE(docT.subCycleIndex() == 0);
        

        SECTION("usage")
        {
            auto& Told = fvcc::oldTime(T);

            REQUIRE(Told.name == "T_0");
            REQUIRE(Told.internalField().copyToHost()[0] == 1.0);
            fvcc::FieldDocument& docTold = fieldCollection.fieldDoc(Told.key);
            REQUIRE(docTold.timeIndex() == 0);
            fvcc::OldTimeCollection& oldTimeCollection = fvcc::OldTimeCollection::instance(fieldCollection);
            REQUIRE(oldTimeCollection.size() == 1);
            auto& ToldDoc = oldTimeCollection.oldTimeDoc(oldTimeCollection.findNextTime(T.key));
            REQUIRE(ToldDoc.nextTime() == T.key);
            REQUIRE(ToldDoc.previousTime() == Told.key);
            REQUIRE(ToldDoc.currentTime() == T.key);
            REQUIRE(ToldDoc.level() == 1);

            auto& sameTold = fvcc::oldTime(T);
            // check if the same field is returned
            REQUIRE(&Told == &sameTold);

            auto& Told2 = fvcc::oldTime(Told);
            REQUIRE(Told2.name == "T_0_0");
            REQUIRE(Told2.internalField().copyToHost()[0] == 1.0);
            fvcc::FieldDocument& docTold2 = fieldCollection.fieldDoc(Told2.key);
            REQUIRE(docTold2.timeIndex() == -1);
            REQUIRE(oldTimeCollection.size() == 2);
            auto& ToldDoc2 = oldTimeCollection.oldTimeDoc(oldTimeCollection.findNextTime(Told.key));
            REQUIRE(ToldDoc2.nextTime() == Told.key);
            REQUIRE(ToldDoc2.previousTime() == Told2.key);
            REQUIRE(ToldDoc2.currentTime() == T.key);
            REQUIRE(ToldDoc2.level() == 2);

            auto& sameTold2 = fvcc::oldTime(Told);
            // check if the same field is returned
            REQUIRE(&Told2 == &sameTold2);
        }
    }
}
