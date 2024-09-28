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

TEST_CASE("Document")
{

    SECTION("create empty document")
    {
        NeoFOAM::Document doc;
        REQUIRE(doc.keys().size() == 1);
    }

    SECTION("create document")
    {
        NeoFOAM::Document doc({{"key1", std::string("value1")}, {"key2", 2.0}});
        REQUIRE(doc.keys().size() == 3);
        REQUIRE(doc.get<std::string>("key1") == "value1");
        REQUIRE(doc.get<double>("key2") == 2.0);

        SECTION("insert values")
        {
            doc.insert("key3", std::string("value3"));
            doc.insert("key4", 4.0);
            REQUIRE(doc.keys().size() == 5);
            REQUIRE(doc.get<std::string>("key3") == "value3");
            REQUIRE(doc.get<double>("key4") == 4.0);
        }
    }

    SECTION("custom validator")
    {
        auto validator = [](const NeoFOAM::Dictionary& dict) {
            return dict.contains("key1") && dict.contains("key2");
         };
        NeoFOAM::Document doc(
            {{"key1", std::string("value1")}, {"key2", 2.0}},
            validator
        );
        REQUIRE(doc.keys().size() == 3);
        REQUIRE(doc.get<std::string>("key1") == "value1");
        REQUIRE(doc.get<double>("key2") == 2.0);
    }
}

TEST_CASE("Database") {
    NeoFOAM::Database db;

    SECTION("createCollection") {
        db.createCollection("collection1");
        db.createCollection("collection2");
        REQUIRE(db.getCollection("collection1").has_value());
        REQUIRE(db.getCollection("collection2").has_value());
        REQUIRE_FALSE(db.getCollection("collection3").has_value());
    }

    SECTION("getCollection") {
        db.createCollection("collection1");

        auto collection1 = db.getCollection("collection1");

        REQUIRE(collection1.has_value());
        REQUIRE(collection1->size() == 0);

        NeoFOAM::Document doc;
        doc.insert("key1", std::string("value1"));
        auto doc1Id = collection1->insert(doc);

        auto retrievedDoc = collection1->getDocument(doc1Id);
        REQUIRE(retrievedDoc.has_value());
        REQUIRE(retrievedDoc->get<std::string>("key1")== "value1");

        REQUIRE(collection1->size() == 1);
    }

    SECTION("queryDocuments") {
        db.createCollection("collection1");

        auto collection1 = db.getCollection("collection1");

        REQUIRE(collection1.has_value());

        NeoFOAM::Document doc1;
        doc1.insert("key1", std::string("value1"));
        collection1->insert(doc1);

        NeoFOAM::Document doc2;
        doc2.insert("key1", std::string("value2"));
        collection1->insert(doc2);

        NeoFOAM::Document doc3;
        doc3.insert("key2", std::string("value3"));
        collection1->insert(doc3);

        auto results = collection1->find([](const NeoFOAM::Document& doc) {
            return doc.contains("key1") && doc.get<std::string>("key1") == "value2";
        });

        REQUIRE(results.size() == 1);
        REQUIRE(results[0].get<std::string>("key1") == "value2");
    }
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
        // auto vf = createVolumeField(mesh);
        // NeoFOAM::fill(vf.internalField(), 1.0);

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
