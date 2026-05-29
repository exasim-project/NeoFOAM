// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/catch_approx.hpp>

#include <cstdlib>
#include <filesystem>

#include "NeoFOAM/NeoFOAM.hpp"

#include "argList.H"
#include "fvMesh.H"
#include "Time.H"

Foam::Time* timePtr;    // A single time object
Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
Foam::fvMesh* meshPtr;  // A single mesh object

int main(int argc, char* argv[])
{
    std::cout << __FILE__ << ":" << __LINE__ << "\n";
    int result;
    NeoN::initialize(argc, argv);
    {
        Catch::Session session;

        // Specify command line options
        int returnCode = session.applyCommandLine(argc, argv);
        if (returnCode != 0) // Indicates a command line error
            return returnCode;

        // Find position of separator "---"
        int sepIdx = argc - 1;
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i], "---") == 0) sepIdx = i;
        }

        // Figure out argc for each part
        int doctestArgc = (sepIdx == argc - 1) ? argc : sepIdx;
        int foamArgc = (sepIdx == argc - 1) ? 1 : argc - sepIdx;

        // Prepare argv for doctestArgv
        char* doctestArgv[doctestArgc];
        for (int i = 0; i < doctestArgc; i++)
        {
            doctestArgv[i] = argv[i];
        }

        // Prepare argv for OpenFOAM
        char* foamArgv[foamArgc];
        foamArgv[0] = argv[0];
        for (int i = 1; i < foamArgc; i++)
        {
            foamArgv[i] = argv[doctestArgc + i];
        }

        // Overwrite argv and argc for Foam include files
        argc = foamArgc;
        for (int i = 1; i < foamArgc; i++)
        {
            argv[i] = foamArgv[i];
        }

        // Generate the polyMesh on the fly when a case directory ships only
        // a blockMeshDict — keeps generated mesh files out of the source tree.
        if (!std::filesystem::exists("constant/polyMesh/points")
            && std::filesystem::exists("system/blockMeshDict"))
        {
            std::cout << "polyMesh not found — running blockMesh...\n";
            int rc = std::system("blockMesh > log.blockMesh 2>&1");
            if (rc != 0)
            {
                std::cerr << "blockMesh failed (rc=" << rc
                          << "); ensure OpenFOAM is sourced and 'blockMesh' is on PATH. "
                          << "See log.blockMesh for details.\n";
                return 1;
            }
            int rc2 = std::system("decomposePar > log.decomposePar 2>&1");
            if (rc2 != 0)
            {
                std::cerr << "decomposePar failed (rc=" << rc2
                          << "); ensure OpenFOAM is sourced and 'decomposePar' is on PATH. "
                          << "See log.decomposePar for details.\n";
                return 1;
            }
        }

#include "setRootCase.H"
#include "createTime.H"

        std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
        Foam::fvMesh& mesh = *meshPtr;
        argsPtr = &args;
        timePtr = &runTime;

        // Run benchmarks if there are any
        result = session.run();
    }
    NeoN::finalize();
    return result;
}
