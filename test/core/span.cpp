// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <limits>
#include <Kokkos_Core.hpp>

#include "NeoFOAM/NeoFOAM.hpp"


TEST_CASE("parallelFor")
{
    NeoFOAM::Executor exec = NeoFOAM::SerialExecutor {};
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
    NeoFOAM::fill(fieldA, 2.0);

    SECTION("parallelFor_") {}
};
