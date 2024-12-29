// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core.hpp"


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
