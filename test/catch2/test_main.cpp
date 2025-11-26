// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors


#include "Kokkos_Core.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/catch_approx.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

#include "argList.H"
#include "fvMesh.H"
#include "Time.H"

Foam::Time* timePtr;    // A single time object
Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
Foam::fvMesh* meshPtr;  // A single mesh object

int main(int argc, char* argv[])
{
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

#include "setRootCase.H"
#include "createTime.H"

        std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
        Foam::fvMesh& mesh = *meshPtr;
        argsPtr = &args;
        timePtr = &runTime;

        int result = session.run();

        // Run benchmarks if there are any
        Kokkos::finalize();

        return result;
    }
    }
