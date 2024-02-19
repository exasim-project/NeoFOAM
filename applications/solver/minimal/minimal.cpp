// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#include <iostream>

#include "NeoFOAM/NeoFOAM.hpp"

int main(int argc, char *argv[]) {
#include "setRootCase.H"
#include "createTime.H"

  Info << "\nStarting time loop\n" << endl;

  while (runTime.loop()) {
    Info << "Time = " << runTime.timeName() << nl << endl;
  }

  runTime.printExecutionTime(Info);

  Info << "End\n" << endl;

  return 0;
}
