// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include <Kokkos_Core.hpp>

#include "NeoFOAM/core/mpi/halfDuplexCommBuffer.hpp"
#include "NeoFOAM/core/mpi/environment.hpp"

using namespace NeoFOAM;
using namespace NeoFOAM::mpi;

TEST_CASE("halfDuplexBuffer")
{

    MPIEnvironment mpiEnviron;
    std::vector<std::size_t> rankCommSize(mpiEnviron.sizeRank(), 1);
    HalfDuplexCommBuffer<Kokkos::HostSpace> buffer(mpiEnviron, rankCommSize);

    SECTION("Default Constructor")
    {
        HalfDuplexCommBuffer buffer2;
        REQUIRE_FALSE(buffer2.isCommInit());
    }

    SECTION("Parameterized Constructor") { REQUIRE_FALSE(buffer.isCommInit()); }

    SECTION("Init and Finalise")
    {
        REQUIRE(false == buffer.isCommInit());
        REQUIRE("unassigned" == buffer.getCommName());
        buffer.initComm<int>("Init Comm");
        REQUIRE(true == buffer.isCommInit());
        REQUIRE(false == buffer.isActive());
        REQUIRE("Init Comm" == buffer.getCommName());
        buffer.finaliseComm();
        REQUIRE("unassigned" == buffer.getCommName());
        REQUIRE(false == buffer.isCommInit());
    }

    SECTION("Set Comm Rank Size")
    {
        for (int rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
            rankCommSize[rank] = rank;
        buffer.setCommRankSize<double>(rankCommSize);
        buffer.initComm<double>("Set Comm Rank Size");
        for (int rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = buffer.get<double>(rank);
            REQUIRE(data.size() == rank);
        }
        buffer.finaliseComm();
    }

    SECTION("Send and Receive")
    {

        HalfDuplexCommBuffer send(mpiEnviron, rankCommSize);
        HalfDuplexCommBuffer receive(mpiEnviron, rankCommSize);

        send.initComm<int>("Send and Receive");
        receive.initComm<int>("Send and Receive");
        for (int rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = send.get<int>(rank);
            data[0] = rank;
        }

        REQUIRE(false == send.isActive());
        REQUIRE(false == receive.isActive());

        send.send();
        receive.receive();

        // we cant test isActive is true here because it becomes a race condition.

        send.waitComplete();
        receive.waitComplete();

        REQUIRE(false == send.isActive());
        REQUIRE(false == receive.isActive());

        for (int rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = receive.get<int>(rank);
            REQUIRE(data[0] == mpiEnviron.rank());
        }

        send.finaliseComm();
        receive.finaliseComm();
    }
}
