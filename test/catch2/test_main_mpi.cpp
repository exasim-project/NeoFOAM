// SPDX-FileCopyrightTex#include "fvCFD.H"t: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_adapters.hpp"
#include "catch2/reporters/catch_reporter_registrars.hpp"
#include "Kokkos_Core.hpp"
#include "NeoN/core/initialization.hpp"
#include "NeoN/core/mpi/environment.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

#include "mpiReporter.hpp"
#include "mpiSerialization.hpp"

#include "fvCFD.H"
#include "argList.H"
#include "fvMesh.H"
#include "Time.H"

#include <cstdlib>
#include <filesystem>
#include <iostream>

Foam::Time* timePtr;
Foam::argList* argsPtr;
Foam::fvMesh* meshPtr;

CATCH_REGISTER_REPORTER("mpi", MpiReporter);

int main(int argc, char* argv[])
{
    // Init MPI before OpenFOAM's argList does (in setRootCase.H), so rank 0 can
    // auto-generate the polyMesh and the processor decomposition before any rank
    // chdir's into its processor<N>/ directory. argList::checkParallel skips its
    // own MPI_Init when MPI_Initialized() is true, so calling here is safe.
    {
        int alreadyInit = 0;
        MPI_Initialized(&alreadyInit);
        if (!alreadyInit)
        {
            int provided = 0;
            MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        }
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0)
        {
            if (!std::filesystem::exists("constant/polyMesh/points")
                && std::filesystem::exists("system/blockMeshDict"))
            {
                std::cout << "polyMesh not found — running blockMesh...\n";
                int rc = std::system("blockMesh > log.blockMesh 2>&1");
                if (rc != 0)
                {
                    std::cerr << "blockMesh failed (rc=" << rc << "); see log.blockMesh\n";
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }

            if (std::filesystem::exists("system/decomposeParDict")
                && !std::filesystem::exists("processor0/constant/polyMesh/points"))
            {
                std::cout << "processor decomposition not found — running decomposePar...\n";
                int rc = std::system("decomposePar -force > log.decomposePar 2>&1");
                if (rc != 0)
                {
                    std::cerr << "decomposePar failed (rc=" << rc << "); see log.decomposePar\n";
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // create a thread (on the root process) that serializes the IO
    bool threadShutdown = false;
    std::thread sequalizeIOThread {serializeIO, &threadShutdown};

    // Initialize Catch2
    int result;
    // ensure any kokkos initialization output will appear first
    std::cout << std::flush;
    std::cerr << std::flush;

    int sepIdx = argc - 1;

    // Figure out argc for each part
    int doctestArgc = (sepIdx == argc - 1) ? argc : sepIdx;
    int foamArgc = doctestArgc; //(sepIdx == argc - 1) ? 1 : argc - sepIdx;

    // Prepare argv for doctestArgv
    char* doctestArgv[doctestArgc];
    for (int i = 0; i < doctestArgc; i++)
    {
        doctestArgv[i] = argv[i];
    }

    // Prepare argv for OpenFOAM
    char* foamArgv[foamArgc];
    Catch::Session session;
#include "setRootCase.H"
#include "createTime.H"

    NeoN::initialize(argc, argv);
    {
        std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
        Foam::fvMesh& mesh = *meshPtr;
        argsPtr = &args;
        timePtr = &runTime;

        result = session.run();
        MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, COMM);

        MPI_Barrier(COMM);
        threadShutdown = true;
        sequalizeIOThread.join();
    }
    NeoN::finalize();

    return result;
}
