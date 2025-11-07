// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

Info << "Reading transportProperties\n" << endl;

Foam::IOdictionary transportProperties(Foam::IOobject(
    "transportProperties",
    runTime.constant(),
    mesh,
    Foam::IOobject::MUST_READ_IF_MODIFIED,
    Foam::IOobject::NO_WRITE
));


Info << "Reading field T\n" << endl;
Foam::volScalarField
    T(Foam::IOobject(
          "T",
          runTime.timeName(),
          mesh,
          Foam::IOobject::MUST_READ,
          Foam::IOobject::AUTO_WRITE
      ),
      mesh);


Info << "Reading field kappa\n" << endl;
Foam::surfaceScalarField kappa(
    Foam::IOobject(
        "kappa",
        runTime.timeName(),
        mesh,
        Foam::IOobject::NO_READ,
        Foam::IOobject::NO_WRITE
    ),
    mesh,
    Foam::dimensionedScalar("kappa", Foam::dimViscosity, transportProperties)
);
