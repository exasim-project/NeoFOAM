// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

Info << "Reading field p\n" << endl;
Foam::volScalarField
    ofP(Foam::IOobject(
            "p",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        mesh);


Info << "Reading field U\n" << endl;
Foam::volVectorField
    ofU(Foam::IOobject(
            "U",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        mesh);


Info << "Reading/calculating face flux field phi\n" << endl;

Foam::surfaceScalarField ofPhi(
    Foam::IOobject(
        "phi",
        runTime.timeName(),
        mesh,
        Foam::IOobject::READ_IF_PRESENT,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ),
    fvc::flux(ofU)
);

Foam::singlePhaseTransportModel laminarTransport(ofU, ofPhi);
auto tnu = laminarTransport.nu();

Foam::label pRefCell = 0;
Foam::scalar pRefValue = 0.0;
setRefCell(ofP, mesh.solutionDict().subDict("PIMPLE"), pRefCell, pRefValue);
