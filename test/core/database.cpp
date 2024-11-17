// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core/database.hpp"
#include "NeoFOAM/core/collection.hpp"
#include "NeoFOAM/core/document.hpp"

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
        REQUIRE(doc.id().substr(0, 4) == "doc_");
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

        SECTION("valid document")
        {
            NeoFOAM::Document doc({{"key1", std::string("value1")}, {"key2", 2.0}}, validator);
            REQUIRE_NOTHROW(doc.validate());
            REQUIRE(doc.keys().size() == 3);
            REQUIRE(doc.get<std::string>("key1") == "value1");
            REQUIRE(doc.get<double>("key2") == 2.0);
        }

        SECTION("invalid document")
        {
            REQUIRE_THROWS(
                NeoFOAM::Document({{"key1", std::string("value1")}}, validator).validate()
            );
        }
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

        auto& collection1 = db.getCollection("collection1");

        SECTION("check database access from collection")
        {
            REQUIRE(&db == &collection1.db());
        }

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

        auto& collection1 = db.getCollection("collection1");

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

    SECTION("custom Document")
    {
        auto doc = CustomTestDocument::create("name1", 1.0);
        REQUIRE(name(doc) == "name1");
        REQUIRE(value(doc) == 1.0);
    }

    SECTION("custom Collection")
    {
        CustomTestCollection::create("customCollection", db);

        NeoFOAM::Collection& customCollection =
            CustomTestCollection::getCollection(db, "customCollection");

        REQUIRE(customCollection.name() == "customCollection");
        REQUIRE(customCollection.size() == 0);

        SECTION("insert document")
        {
            auto doc1 = CustomTestDocument::create("name1", 1.0);
            REQUIRE(doc1.id().substr(0, 4) == "doc_");
            auto keyDoc1 = customCollection.insert(doc1);

            REQUIRE(customCollection.size() == 1);

            SECTION("get existing document")
            {
                auto& testDoc = customCollection.get(keyDoc1);

                REQUIRE(name(testDoc) == "name1");
                REQUIRE(value(testDoc) == 1.0);
            }

            SECTION("modify document")
            {
                auto& testDoc = customCollection.get(keyDoc1);
                value(testDoc) = 3.0;

                auto& testDoc2 = customCollection.get(keyDoc1);
                REQUIRE(value(testDoc2) == 3.0);

                // update without id
                value(testDoc) = 4.0;

                auto& testDoc3 = customCollection.get(keyDoc1);
                REQUIRE(value(testDoc3) == 4.0);
            }

            SECTION("get non existing document")
            {
                REQUIRE_THROWS(customCollection.get("doesNotExcist"));
            }

            SECTION("query existing document")
            {
                // query documents
                auto results =
                    customCollection.find([](const NeoFOAM::Document& doc)
                                          { return doc.contains("name") && name(doc) == "name1"; });

                REQUIRE(results.size() == 1);
            }

            SECTION("query non existing document")
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
}
