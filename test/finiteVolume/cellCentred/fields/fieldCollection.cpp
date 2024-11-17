// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/fieldCollection.hpp"
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
        NeoFOAM::Field internalField = NeoFOAM::Field<NeoFOAM::scalar>(mesh.exec(), mesh.nCells(), 1.0);
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


TEST_CASE("Field Document")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );


    std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("create FieldDocument: " + execName)
    {
        auto doc = fvcc::FieldDocument::create(
            {.name = "T",
             .timeIndex = 1,
             .iterationIndex = 2,
             .subCycleIndex = 3,
             .field = createVolumeField(mesh, "T")}
        );

        SECTION("validate FieldDocument")
        {
            REQUIRE_NOTHROW(doc.validate());
            REQUIRE(doc.keys().size() == 6);
            REQUIRE(doc.id().substr(0, 4) == "doc_");
            REQUIRE(name(doc) == "T");
            const auto& volField = fvcc::volField<NeoFOAM::scalar>(doc);

            REQUIRE(volField.name == "T");
            REQUIRE(volField.internalField().copyToHost()[0] == 1.0);
            REQUIRE(fvcc::timeIndex(doc) == 1);
            REQUIRE(fvcc::iterationIndex(doc) == 2);
            REQUIRE(fvcc::subCycleIndex(doc) == 3);
        }

        SECTION("modify fieldDocument")
        {
            fvcc::timeIndex(doc) = 4;
            fvcc::iterationIndex(doc) = 5;
            fvcc::subCycleIndex(doc) = 6;
            auto& volField = fvcc::volField<NeoFOAM::scalar>(doc);
            NeoFOAM::fill(volField.internalField(), 2.0);

            REQUIRE(volField.internalField().copyToHost()[0] == 2.0);
            REQUIRE(fvcc::timeIndex(doc) == 4);
            REQUIRE(fvcc::iterationIndex(doc) == 5);
            REQUIRE(fvcc::subCycleIndex(doc) == 6);
        }
    }
}

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

    SECTION("create FieldCollection: " + execName)
    {
        fvcc::FieldCollection::create(db, "testFieldCollection");

        auto& fieldCollection = fvcc::FieldCollection::getCollection(db, "testFieldCollection");
        REQUIRE(fieldCollection.size() == 0);
    }

    SECTION("add FieldDocument to FieldCollection")
    {
        fvcc::FieldCollection::create(db, "testFieldCollection");

        auto& fieldCollection = fvcc::FieldCollection::getCollection(db, "testFieldCollection");
        auto doc1 = fvcc::FieldDocument::create(
            {.name = "T1",
             .timeIndex = 1,
             .iterationIndex = 1,
             .subCycleIndex = 1,
             .field = createVolumeField(mesh, "T1")}
        );

        std::string keyDoc1 = fieldCollection.insert(doc1);

        auto doc2 = fvcc::FieldDocument::create(
            {.name = "T2",
             .timeIndex = 2,
             .iterationIndex = 2,
             .subCycleIndex = 2,
             .field = createVolumeField(mesh, "T2")}
        );

        std::string keyDoc2 = fieldCollection.insert(doc2);

        auto doc3 = fvcc::FieldDocument::create(
            {.name = "T3",
             .timeIndex = 3,
             .iterationIndex = 3,
             .subCycleIndex = 3,
             .field = createVolumeField(mesh, "T3")}
        );

        std::string keyDoc3 = fieldCollection.insert(doc3);

        REQUIRE(fieldCollection.size() == 3);

        SECTION("query FieldDocument from FieldCollection")
        {
            SECTION("query by timeIndex")
            {
                auto resTimeIndex = fieldCollection.find([](const NeoFOAM::Document& doc)
                                                         { return fvcc::timeIndex(doc) == 2; });

                REQUIRE(resTimeIndex.size() == 1);

                const auto& doc = fieldCollection.get(resTimeIndex[0]);
                REQUIRE(name(doc) == "T2");
            }

            SECTION("query by name")
            {
                auto resName = fieldCollection.find([](const NeoFOAM::Document& doc)
                                                    { return name(doc) == "T3"; });

                REQUIRE(resName.size() == 1);

                const auto& doc2 = fieldCollection.get(resName[0]);
                REQUIRE(fvcc::timeIndex(doc2) == 3);
            }
        }
    }

    SECTION("usage")
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
    }
}
