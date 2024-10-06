// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

// #include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
// #include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fieldCollection.hpp"

#include "NeoFOAM/core/database.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

TEST_CASE("Field Document")
{

    SECTION("create FieldDocument")
    {
        auto doc = fvcc::FieldDocument::create({
            .name = "T",
            .timeIndex = 1,
            .iterationIndex = 2,
            .subCycleIndex = 3
        });
        REQUIRE(doc.keys().size() == 5);
        REQUIRE(doc.id().substr(0,4) == "doc_");
        REQUIRE(name(doc) == "T");
        REQUIRE(fvcc::timeIndex(doc) == 1);
        REQUIRE(fvcc::iterationIndex(doc) == 2);
        REQUIRE(fvcc::subCycleIndex(doc) == 3);

        SECTION("modify FieldDocument")
        {
            fvcc::timeIndex(doc) = 4;
            fvcc::iterationIndex(doc) = 5;
            fvcc::subCycleIndex(doc) = 6;
            REQUIRE(fvcc::timeIndex(doc) == 4);
            REQUIRE(fvcc::iterationIndex(doc) == 5);
            REQUIRE(fvcc::subCycleIndex(doc) == 6);
        }
    }
}

