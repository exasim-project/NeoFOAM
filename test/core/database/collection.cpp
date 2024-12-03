// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core/database/database.hpp"
#include "NeoFOAM/core/database/collection.hpp"
#include "NeoFOAM/core/database/document.hpp"

#include "customTestCollection.hpp"

TEST_CASE("CustomCollection")
{
    NeoFOAM::Database db;
    NeoFOAM::Collection collection = CustomCollection(db, "testCollection");
    CustomCollection& customCollection = collection.as<CustomCollection>();

    REQUIRE(customCollection.size() == 0);
    REQUIRE(customCollection.name() == "testCollection");
    REQUIRE(customCollection.type() == "CustomDocument");

    CustomCollection& customCollection2 = CustomCollection::instance(db, "testCollection");

    CustomCollection& customCollection3 = db.at<CustomCollection>("testCollection");

    REQUIRE(&customCollection3 == &customCollection2);

    SECTION("new collection")
    {
        CustomCollection& newCollection = CustomCollection::instance(db, "newCollection");

        REQUIRE(db.size() == 2);
    }
}

TEST_CASE("CustomDocument")
{
    CustomDocument doc;
    REQUIRE(doc.id().find("doc_") != std::string::npos);
    REQUIRE(doc.typeName() == "CustomDocument");

    REQUIRE(doc.doc().id().find("doc_") != std::string::npos);

    CustomDocument doc2(NeoFOAM::Document({{"name", std::string("doc2")}, {"testValue", 42}}));


    REQUIRE(doc2.doc().id().find("doc_") != std::string::npos);
    REQUIRE(doc2.typeName() == "CustomDocument");
    REQUIRE(doc2.doc().get<std::string>("name") == "doc2");
    REQUIRE(doc2.doc().get<int>("testValue") == 42);
    CustomDocument doc3 = CustomDocument("doc3", 3.14);
    REQUIRE(doc3.name() == "doc3");
    REQUIRE(doc3.testValue() == 3.14);

}
