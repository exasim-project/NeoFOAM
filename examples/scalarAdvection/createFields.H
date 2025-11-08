// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

Foam::Info << "Reading field T\n" << Foam::endl;

Foam::volScalarField
    T(Foam::IOobject(
          "T",
          runTime.timeName(),
          mesh,
          Foam::IOobject::MUST_READ,
          Foam::IOobject::AUTO_WRITE
      ),
      mesh);


Foam::volVectorField
    U(Foam::IOobject(
          "U",
          runTime.timeName(),
          mesh,
          Foam::IOobject::MUST_READ,
          Foam::IOobject::AUTO_WRITE
      ),
      mesh);

if (rt.controlDict.get<int>("setFields"))
{
    Foam::scalar spread = 0.05;
    Foam::scalar pi = Foam::constant::mathematical::pi;
    forAll(U, celli)
    {
        // initialize U
        Foam::scalar x = mesh.C()[celli].x();
        Foam::scalar y = mesh.C()[celli].y();

        U[celli].x() = -Foam::sin(2.0 * pi * y) * Foam::pow(Foam::sin(pi * x), 2.0);
        U[celli].y() = Foam::sin(2.0 * pi * x) * Foam::pow(Foam::sin(pi * y), 2.0);
        U[celli].z() = 0.0;

        // initialize T
        T[celli] = std::exp(
            -0.5
            * (std::pow((mesh.C()[celli].x() - 0.5) / spread, 2.0)
               + std::pow((mesh.C()[celli].y() - 0.75) / spread, 2.0))
        );
    }
    T.correctBoundaryConditions();
    T.write();
}

Foam::surfaceScalarField
    phi(Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::READ_IF_PRESENT,
            Foam::IOobject::AUTO_WRITE
        ),
        Foam::linearInterpolate(U) & mesh.Sf());

// Copies of initial U and phi for use when flow is periodic
Foam::surfaceScalarField phi0 = phi;
Foam::volVectorField U0 = U;
