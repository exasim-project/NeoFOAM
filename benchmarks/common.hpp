// SPDX-FileCopyrightText: 2025 NeoFOAM authors
// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-License-Identifier: MIT

void randomizeField(auto& field)
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 2.0);

    forAll(field, celli)
    {
        field[celli] = rand();
    }

    field.correctBoundaryConditions();
}


/* function to create a volScalarField filled with random values for test purposes */
auto randomScalarField(const Foam::fvMesh& mesh, Foam::word name)
{
    Foam::volScalarField t(
        Foam::IOobject(
            name,
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh
    );

    randomizeField(t);
    return t;
}


/**@brief create a dimensionedScalarField with random entries*/
template<typename FieldType>
auto randDimScalarField(const Foam::fvMesh& mesh, Foam::dimensionSet dimensionSet, Foam::word name)
{
    auto field = FieldType(
        Foam::IOobject(
            name,
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar(name, dimensionSet, 0.0)
    );

    randomizeField(field);
    return field;
}
