// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

template<typename FieldType, typename RandomFunc>
FieldType createRandomField(
    const Foam::Time& runTime,
    const Foam::fvMesh& mesh,
    Foam::word name,
    RandomFunc rand
)
{
    FieldType t(
        Foam::IOobject(
            name,
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh
    );

    forAll(t, celli)
    {
        t[celli] = rand();
    }

    t.correctBoundaryConditions();
    return t;
}

/* function to create a volScalarField filled with random values for test purposes */
auto randomScalarField(const Foam::Time& runTime, const Foam::fvMesh& mesh, Foam::word name)
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 2.0);
    return createRandomField<Foam::volScalarField>(runTime, mesh, name, [&]() { return dis(gen); });
}
