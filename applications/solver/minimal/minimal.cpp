/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
\*---------------------------------------------------------------------------*/
#include <iostream>

import NeoFOAM;


int main(int argc, char *argv[])
{
//   #include "setRootCase.H"
     #include "createTime.H"
     std::cout << time << "\n";
//
//
//   Info<< nl;
//   runTime.printExecutionTime(Info);
//
//   Info<< "End\n" << endl;

    return 0;
}
