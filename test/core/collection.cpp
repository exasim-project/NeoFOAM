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

class CustomDocument
{
public:

    NeoFOAM::Document& doc() { return doc_; }

    const NeoFOAM::Document& doc() const { return doc_; }

    std::string id() const { return doc_.id(); }

    static std::string typeName() { return "CustomDocument"; }

private:

    NeoFOAM::Document doc_;
};

class CustomCollection : public NeoFOAM::CollectionMixin<CustomDocument>
{
public:

    CustomCollection(NeoFOAM::Database& db, std::string name)
        : NeoFOAM::CollectionMixin<CustomDocument>(db, name)
    {}
};


TEST_CASE("Collection")
{
    NeoFOAM::Database db;
    NeoFOAM::Collection collection = CustomCollection(db, "testCollection");
    CustomCollection& customCollection = collection.as<CustomCollection>();

    REQUIRE(customCollection.size() == 0);
    REQUIRE(customCollection.name() == "testCollection");
    REQUIRE(customCollection.type() == "CustomDocument");

    CustomCollection& customCollection2 = db.get<CustomCollection>("testCollection");

    // SECTION("get")
    // {
    //     auto& doc = customCollection.get("doc1");
    //     REQUIRE(doc.id() == "doc1");
    // }
}
