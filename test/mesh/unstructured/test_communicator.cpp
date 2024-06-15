// SPDX-License-Identifier: Unlicense
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include "NeoFOAM/mesh/unstructured/communicator.hpp"

using namespace NeoFOAM;
using namespace NeoFOAM::mpi;

TEST_CASE("CommMap Initialization")
{

    Communicator comm;

    SECTION("Default constructor") {}
}
