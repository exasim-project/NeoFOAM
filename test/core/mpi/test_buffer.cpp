// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core/mpi/buffer.hpp"

using namespace NeoFOAM;
using namespace NeoFOAM::mpi;

TEST_CASE("Buffer")
{
    Buffer buffer;
    REQUIRE_NOTHROW(buffer.setCommTypeSize<int>({1, 2, 3}));
}

TEST_CASE("BufferSizing")
{
    Buffer buffer;
    std::vector<std::size_t> rankComm = {2, 3, 1};

    SECTION("Int")
    {
        buffer.setCommTypeSize<int>(rankComm);
        auto span = buffer.get<int>(0);
        REQUIRE(span.size() == 2);

        span = buffer.get<int>(1);
        REQUIRE(span.size() == 3);

        span = buffer.get<int>(2);
        REQUIRE(span.size() == 1);
    }

    SECTION("Double")
    {
        buffer.setCommTypeSize<double>(rankComm);
        auto span = buffer.get<double>(0);
        REQUIRE(span.size() == 2);

        span = buffer.get<double>(1);
        REQUIRE(span.size() == 3);

        span = buffer.get<double>(2);
        REQUIRE(span.size() == 1);
    }

    SECTION("OnlySizeUp")
    {
        Buffer buffer;
        std::vector<std::size_t> rankComm1 = {2, 3, 1};
        buffer.setCommTypeSize<int>(rankComm1);

        auto previousSize = buffer.get<int>(0).size() * sizeof(int)
                          + buffer.get<int>(1).size() * sizeof(int)
                          + buffer.get<int>(2).size() * sizeof(int);

        std::vector<std::size_t> rankComm2 = {1, 1, 1};
        buffer.setCommTypeSize<int>(rankComm2);

        auto currentSize = buffer.get<int>(0).size() * sizeof(int)
                         + buffer.get<int>(1).size() * sizeof(int)
                         + buffer.get<int>(2).size() * sizeof(int);

        REQUIRE(currentSize <= previousSize);
    }
}

TEST_CASE("Read/Write")
{
    Buffer buffer;
    std::vector<std::size_t> rankComm = {2, 3, 1};
    buffer.setCommTypeSize<int>(rankComm);

    auto span = buffer.get<int>(0);
    std::fill(span.begin(), span.end(), 1);
    span = buffer.get<int>(1);
    std::fill(span.begin(), span.end(), 2);
    span = buffer.get<int>(2);
    std::fill(span.begin(), span.end(), 3);

    SECTION("Rank0")
    {
        auto span = buffer.get<int>(0);
        REQUIRE(span.size() == 2);
        REQUIRE(span[0] == 1);
        REQUIRE(span[1] == 1);
    }

    SECTION("Rank1")
    {
        auto span = buffer.get<int>(1);
        REQUIRE(span.size() == 3);
        REQUIRE(span[0] == 2);
        REQUIRE(span[1] == 2);
        REQUIRE(span[2] == 2);
    }

    SECTION("Rank2")
    {
        auto span = buffer.get<int>(2);
        REQUIRE(span.size() == 1);
        REQUIRE(span[0] == 3);
    }
}
