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


TEST_CASE("Field Collection")
{
    NeoFOAM::Database db;

    SECTION("create FieldCollection")
    {
        // fvcc::FieldCollection::registerCollection("fieldCollection", db);
        // fvcc::FieldCollection fieldCollection("fieldCollection", db);
        // REQUIRE(fieldCollection.name() == "fieldCollection");
        // REQUIRE(fieldCollection.size() == 0);

        // SECTION("insert Field")
        // {
        //     auto field = createVolumeField(NeoFOAM::createSingleCellMesh(NeoFOAM::SerialExecutor {}), "T");
        //     fieldCollection.insert(field);
        //     REQUIRE(fieldCollection.size() == 1);
        // }

        // SECTION("get Field")
        // {
        //     auto field = createVolumeField(NeoFOAM::createSingleCellMesh(NeoFOAM::SerialExecutor {}), "T");
        //     auto key = fieldCollection.insert(field);
        //     auto retrievedField = fieldCollection.get(key);
        //     REQUIRE(retrievedField.name() == "T");
        // }

        // SECTION("query Fields")
        // {
        //     auto field1 = createVolumeField(NeoFOAM::createSingleCellMesh(NeoFOAM::SerialExecutor {}), "T");
        //     auto key1 = fieldCollection.insert(field1);

        //     auto field2 = createVolumeField(NeoFOAM::createSingleCellMesh(NeoFOAM::SerialExecutor {}), "T2");
        //     auto key2 = fieldCollection.insert(field2);

        //     auto results = fieldCollection.find([](const fvcc::VolumeField<NeoFOAM::scalar>& field) {
        //         return field.name() == "T2";
        //     });

        //     REQUIRE(results.size() == 1);
        //     REQUIRE(fieldCollection.get(results[0]).name() == "T2");
        // }
    }
}
