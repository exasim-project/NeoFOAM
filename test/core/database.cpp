// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core/database.hpp"

#include "customTestCollection.hpp"


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
        auto validator = [](const NeoFOAM::Dictionary& dict)
        { return dict.contains("key1") && dict.contains("key2"); };
        NeoFOAM::Document doc({{"key1", std::string("value1")}, {"key2", 2.0}}, validator);
        REQUIRE(doc.keys().size() == 3);
        REQUIRE(doc.get<std::string>("key1") == "value1");
        REQUIRE(doc.get<double>("key2") == 2.0);
    }
}


TEST_CASE("Database")
{
    NeoFOAM::Database db;

    SECTION("createCollection")
    {
        db.createCollection("collection1", "testCollection");
        db.createCollection("collection2", "testCollection");
    }

    SECTION("getCollection")
    {
        db.createCollection("collection1", "testCollection");

        auto collection1 = db.getCollection("collection1");

        REQUIRE(collection1.size() == 0);

        SECTION("get non existing collection") { REQUIRE_THROWS(db.getCollection("doesNotExist")); }

        NeoFOAM::Document doc;
        doc.insert("key1", std::string("value1"));
        auto doc1Id = collection1.insert(doc);

        auto& retrievedDoc = collection1.get(doc1Id);
        REQUIRE(retrievedDoc.get<std::string>("key1") == "value1");

        REQUIRE(collection1.size() == 1);
    }

    SECTION("query Documents")
    {
        db.createCollection("collection1", "testCollection");

        auto collection1 = db.getCollection("collection1");

        NeoFOAM::Document doc1;
        doc1.insert("key1", std::string("value1"));
        collection1.insert(doc1);

        NeoFOAM::Document doc2;
        doc2.insert("key1", std::string("value2"));
        collection1.insert(doc2);

        NeoFOAM::Document doc3;
        doc3.insert("key2", std::string("value3"));
        collection1.insert(doc3);

        auto results = collection1.find(
            [](const NeoFOAM::Document& doc)
            { return doc.contains("key1") && doc.get<std::string>("key1") == "value2"; }
        );

        REQUIRE(results.size() == 1);

        REQUIRE(collection1.get(results[0]).get<std::string>("key1") == "value2");
    }

    SECTION("custom Collection")
    {
        CustomTestCollection::registerCollection("customCollection", db);

        auto customCollection = CustomTestCollection::getCollection("customCollection", db);

        REQUIRE(customCollection.name() == "customCollection");
        REQUIRE(customCollection.size() == 0);


        // insert documents
        CustomTestDocument doc1 {"name1", 1.0};
        auto keyDoc1 = customCollection.insert(doc1);

        REQUIRE(customCollection.size() == 1);

        CustomTestDocument doc2 {"name2", 2.0};
        auto keyDoc2 = customCollection.insert(doc2);

        REQUIRE(customCollection.size() == 2);

        SECTION("get existing documents")
        {
            CustomTestDocument testDoc = customCollection.get(keyDoc1);

            REQUIRE(testDoc.name == "name1");
            REQUIRE(testDoc.value == 1.0);
        }

        SECTION("modify document")
        {
            CustomTestDocument testDoc = customCollection.get(keyDoc1);
            testDoc.value = 3.0;
            customCollection.update(keyDoc1, testDoc);

            CustomTestDocument testDoc2 = customCollection.get(keyDoc1);
            REQUIRE(testDoc2.value == 3.0);

            // update without id
            testDoc.value = 4.0;
            customCollection.update(testDoc);

            CustomTestDocument testDoc3 = customCollection.get(keyDoc1);
            REQUIRE(testDoc3.value == 4.0);
        }

        SECTION("get non existing documents") { REQUIRE_THROWS(customCollection.get("id3")); }

        SECTION("query existing documents")
        {
            // query documents
            auto results = customCollection.find(
                [](const NeoFOAM::Document& doc)
                { return doc.contains("name") && doc.get<std::string>("name") == "name2"; }
            );

            REQUIRE(results.size() == 1);
        }

        SECTION("query non existing documents")
        {
            // query documents
            auto results = customCollection.find(
                [](const NeoFOAM::Document& doc) {
                    return doc.contains("notValidKey")
                        && doc.get<std::string>("notValidKey") == "name3";
                }
            );

            REQUIRE(results.size() == 0);
        }
    }
}
