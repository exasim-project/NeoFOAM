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
    std::size_t timeIndex = 0;
    std::size_t iterationIndex = 0;
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
        NeoFOAM::Field internalField =
            NeoFOAM::Field<NeoFOAM::scalar>(mesh.exec(), mesh.nCells(), 1.0);
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

        fvcc::FieldDocument fieldDoc(createVolumeField(mesh, "T"), 1, 2, 3);


        SECTION("validate FieldDocument")
        {
            REQUIRE(fieldDoc.doc().validate());
            REQUIRE_NOTHROW(fieldDoc.doc().validate());

            REQUIRE(fieldDoc.doc().keys().size() == 6);
            REQUIRE(fieldDoc.id().substr(0, 4) == "doc_");
            REQUIRE(fieldDoc.timeIndex() == 1);
            REQUIRE(fieldDoc.iterationIndex() == 2);
            REQUIRE(fieldDoc.subCycleIndex() == 3);
            REQUIRE(fieldDoc.name() == "T");


            NeoFOAM::Document& doc = fieldDoc.doc();
            REQUIRE(doc.validate());
            REQUIRE_NOTHROW(doc.validate());
            REQUIRE(doc.keys().size() == 6);
            REQUIRE(doc.id().substr(0, 4) == "doc_");


            // REQUIRE(name(doc) == "T");
            const auto& constVolField = fieldDoc.field<fvcc::VolumeField<NeoFOAM::scalar>>();
            auto& volField = fieldDoc.field<fvcc::VolumeField<NeoFOAM::scalar>>();

            REQUIRE(volField.name == "T");
            REQUIRE(volField.internalField().copyToHost()[0] == 1.0);
            REQUIRE(&volField == &constVolField);
        }

        SECTION("modify fieldDocument")
        {
            fieldDoc.timeIndex() = 4;
            fieldDoc.iterationIndex() = 5;
            fieldDoc.subCycleIndex() = 6;
            auto& volField = fieldDoc.field<fvcc::VolumeField<NeoFOAM::scalar>>();
            NeoFOAM::fill(volField.internalField(), 2.0);

            REQUIRE(volField.internalField().copyToHost()[0] == 2.0);
            REQUIRE(fieldDoc.timeIndex() == 4);
            REQUIRE(fieldDoc.iterationIndex() == 5);
            REQUIRE(fieldDoc.subCycleIndex() == 6);
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
        fvcc::FieldCollection fieldCollection(db, "testFieldCollection");
        REQUIRE(fieldCollection.size() == 0);
    }

    SECTION("add FieldDocument to FieldCollection" + execName)
    {
        fvcc::FieldCollection fieldCollection =
            fvcc::FieldCollection::instance(db, "testFieldCollection");
        REQUIRE(db.size() == 1);

        fvcc::FieldDocument fieldDoc(createVolumeField(mesh, "T1"), 1, 2, 3);

        REQUIRE(fieldCollection.insert(fieldDoc) != std::string(""));
        REQUIRE(
            fieldCollection.insert(fvcc::FieldDocument(createVolumeField(mesh, "T2"), 1, 2, 3))
            != std::string("")
        );
        REQUIRE(
            fieldCollection.insert(fvcc::FieldDocument(createVolumeField(mesh, "T3"), 1, 2, 3))
            != std::string("")
        );

        REQUIRE(fieldCollection.size() == 3);

        SECTION("get FieldDocument from FieldCollection")
        {
            fvcc::FieldDocument& doc = fieldCollection.fieldDoc(fieldDoc.id());
            REQUIRE(doc.doc().validate());
            REQUIRE(doc.doc().keys().size() == 6);
            REQUIRE(doc.id().substr(0, 4) == "doc_");
            REQUIRE(doc.timeIndex() == 1);
            REQUIRE(doc.iterationIndex() == 2);
            REQUIRE(doc.subCycleIndex() == 3);
            REQUIRE(doc.name() == "T1");

            const auto& constVolField = doc.field<fvcc::VolumeField<NeoFOAM::scalar>>();
            auto& volField = doc.field<fvcc::VolumeField<NeoFOAM::scalar>>();

            REQUIRE(volField.name == "T1");
            REQUIRE(volField.internalField().copyToHost()[0] == 1.0);
            REQUIRE(&volField == &constVolField);
        }

        SECTION("query")
        {
            auto resTimeIndex =
                fieldCollection.find([](const NeoFOAM::Document& doc)
                                     { return doc.get<std::size_t>("timeIndex") == 1; });

            REQUIRE(resTimeIndex.size() == 3);

            auto resSubCycleIndex =
                fieldCollection.find([](const NeoFOAM::Document& doc)
                                     { return doc.get<std::int64_t>("subCycleIndex") == 5; });

            REQUIRE(resSubCycleIndex.size() == 0);

            auto resName = fieldCollection.find([](const NeoFOAM::Document& doc)
                                                { return doc.get<std::string>("name") == "T3"; });

            REQUIRE(resName.size() == 1);

            const auto& fieldDoc2 = fieldCollection.fieldDoc(resName[0]);
            REQUIRE(fieldDoc2.timeIndex() == 1);
        }
    }

    SECTION("register " + execName)
    {

        fvcc::FieldCollection& fieldCollection =
            fvcc::FieldCollection::instance(db, "newTestFieldCollection");
        REQUIRE(db.size() == 1);

        fvcc::VolumeField<NeoFOAM::scalar>& T =
            fieldCollection.registerField<fvcc::VolumeField<NeoFOAM::scalar>>(CreateField {
                .name = "T", .mesh = mesh, .timeIndex = 1, .iterationIndex = 1, .subCycleIndex = 1
            });

        REQUIRE(T.name == "T");
        REQUIRE(T.hasDatabase());
        REQUIRE(T.internalField().copyToHost()[0] == 1.0);
        REQUIRE(T.registered());

        SECTION("Construct from Field")
        {
            fvcc::FieldCollection& fieldCollection = fvcc::FieldCollection::instance(T);
            REQUIRE(fieldCollection.size() == 1);
            const fvcc::VolumeField<NeoFOAM::scalar>& constT = T;
            const fvcc::FieldCollection& fieldCollection3 = fvcc::FieldCollection::instance(constT);
            REQUIRE(fieldCollection3.size() == 1);
        }


        SECTION("register from existing field")
        {
            fvcc::FieldCollection& fieldCollection = fvcc::FieldCollection::instance(T);
            fvcc::VolumeField<NeoFOAM::scalar>& T3 =
                fieldCollection.registerField<fvcc::VolumeField<NeoFOAM::scalar>>(
                    fvcc::CreateFromExistingField {.name = "T3", .field = T}
                );

            const fvcc::FieldDocument& docT = fieldCollection.fieldDoc(T3.key);
            const fvcc::FieldDocument& docT3 = fieldCollection.fieldDoc(T.key);

            REQUIRE(docT.timeIndex() == docT3.timeIndex());
            REQUIRE(docT.iterationIndex() == docT3.iterationIndex());
            REQUIRE(docT.subCycleIndex() == docT3.subCycleIndex());
        }
    }
}
