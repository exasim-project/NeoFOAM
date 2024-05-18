// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#include <iostream>

#include "NeoFOAM/NeoFOAM.hpp"

int main(int argc, char* argv[])
{

#include "setRootCase.H"
#include "createTime.H"

    NF_INFO("Starting time loop");

    while (runTime.loop())
    {
        NF_INFO("Time = " << runTime.timeName() << nl);
    }

    runTime.printExecutionTime(std::cout);

    NF_INFO("End");

    return 0;
}
