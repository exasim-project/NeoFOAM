// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoFOAM/core/mpi/halfDuplexBuffer.hpp"
#include "NeoFOAM/core/mpi/environment.hpp"
#include "NeoFOAM/core/mpi/operators.hpp"
#include <cstring>


using namespace NeoFOAM;
using namespace NeoFOAM::mpi;

TEST_CASE("HalfDuplexBuffer")
{

    MPIEnvironment mpiEnviron;
    std::vector<std::size_t> rankCommSize(mpiEnviron.sizeRank(), 1);
    HalfDuplexBuffer buffer(mpiEnviron, rankCommSize);

    SECTION("Parameterized Constructor")
    {
        HalfDuplexBuffer buffer2;
        REQUIRE_FALSE(buffer2.isCommInit());
    }

    SECTION("Parameterized Constructor") { REQUIRE_FALSE(buffer.isCommInit()); }

    SECTION("Init and Finalise")
    {
        buffer.initComm<int>("Init Comm");
        REQUIRE(buffer.isCommInit());
        REQUIRE(true == buffer.isComplete());
        REQUIRE(buffer.getCommName() == "Init Comm");
        buffer.finaliseComm();
        REQUIRE(buffer.getCommName() == "unassigned");
        REQUIRE(true == buffer.isComplete());
    }

    SECTION("Set Comm Rank Size")
    {
        for (int rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
            rankCommSize[rank] = rank;
        buffer.setCommRankSize<double>({0, 1, 2});
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

        HalfDuplexBuffer send(mpiEnviron, rankCommSize);
        HalfDuplexBuffer receive(mpiEnviron, rankCommSize);

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
