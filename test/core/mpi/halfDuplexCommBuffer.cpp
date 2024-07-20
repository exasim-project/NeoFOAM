// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoFOAM/core/mpi/halfDuplexCommBuffer.hpp"
#include "NeoFOAM/core/mpi/environment.hpp"

using namespace NeoFOAM;
using namespace NeoFOAM::mpi;

TEST_CASE("halfDuplexBuffer")
{

    MPIEnvironment mpiEnviron;
    std::vector<std::size_t> rankCommSize(mpiEnviron.sizeRank(), 1);
    HalfDuplexCommBuffer buffer(mpiEnviron, rankCommSize);

    SECTION("Default Constructor")
    {
        HalfDuplexCommBuffer buffer2;
        REQUIRE_FALSE(buffer2.isCommInit());
    }

    SECTION("Parameterized Constructor") { REQUIRE_FALSE(buffer.isCommInit()); }

    SECTION("Init and Finalise")
    {
        buffer.initComm<int>("Init Comm");
        REQUIRE(buffer.isCommInit());
        REQUIRE(true == buffer.isActive());
        REQUIRE(buffer.getCommName() == "Init Comm");
        buffer.finaliseComm();
        REQUIRE(buffer.getCommName() == "unassigned");
        REQUIRE(!buffer.isCommInit());
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

        send.send();
        receive.receive();

        send.waitComplete();
        receive.waitComplete();

        for (int rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = receive.get<int>(rank);
            REQUIRE(data[0] == mpiEnviron.rank());
        }

        send.finaliseComm();
        receive.finaliseComm();
    }
}
