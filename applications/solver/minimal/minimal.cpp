// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#include "NeoFOAM/NeoFOAM.hpp"

int main(int argc, char* argv[])
{
    auto runTime = NeoFOAM::runTime::initialize(argc, argv);
    auto log = runTime.getLogger();

    while (runTime.loop())
    {
        log.info("Time = {}", runTime.timeName());
    }

    runTime.printExecutionTime();

    runTime.finalize();

    return 0;
}
