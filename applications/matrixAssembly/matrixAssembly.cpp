/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    matrixAssembly


Description


\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "profiling.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{

    #include "addProfilingOption.H"
    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"


    #include "createFields.H"
    runTime.setDeltaT(1.0);

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    ++runTime;

    Info<< "Time = " << runTime.timeName() << nl << endl;
    
    {
        addProfiling(fvScalarMatrix, "TEqnTemporalOperator");
        fvScalarMatrix TEqnTemporalOperator
        (
            fvm::ddt(T)
        );
    }


    {
        addProfiling(fvScalarMatrix, "TEqnDiffusionOperator");
        fvScalarMatrix TEqnDiffusionOperator
        (
            fvm::laplacian(Gamma, T)
        );
    }

    {
        addProfiling(fvScalarMatrix, "TEqnConvectionOperator");
        fvScalarMatrix TEqnConvectionOperator
        (
            fvm::div(phi, T)
        );
    }


    {
        addProfiling(fvScalarMatrix, "EnergyEquation");
        fvScalarMatrix EnergyEquation
        (
            fvm::ddt(rho, T)
            + fvm::div(phi, T)
            - fvm::laplacian(Gamma, T)
        );
    }

    {
        addProfiling(fvScalarMatrix, "NSE");
        fvVectorMatrix NSE
        (
            fvm::ddt(rho, U)
            + fvm::div(phi, U)
            - fvm::laplacian(mu, U)
        );
    }

    profiling::writeNow();
    
    runTime.printExecutionTime(Info);

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
