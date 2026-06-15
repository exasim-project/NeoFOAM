// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

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

Foam::autoPtr<Foam::incompressible::turbulenceModel>
    turbulence(Foam::incompressible::turbulenceModel::New(ofU, ofPhi, laminarTransport));

const Foam::incompressible::LESModel& lesModel =
    Foam::refCast<const Foam::incompressible::LESModel>(turbulence());

Foam::wallDist y(mesh);

// nearWallDist: boundary-face distances from turbulence model (distinct from wallDist)
Foam::volScalarField ofNearWallDist(
    Foam::IOobject(
        "nearWallDist",
        runTime.timeName(),
        mesh,
        Foam::IOobject::NO_READ,
        Foam::IOobject::NO_WRITE
    ),
    mesh,
    Foam::dimensionedScalar("zero", Foam::dimLength, Foam::scalar(0.0))
);
for (Foam::label patchi = 0; patchi < mesh.boundary().size(); ++patchi)
{
    const Foam::scalarField& ofY = turbulence().y()[patchi];
    Foam::fvPatchScalarField& yPatch = ofNearWallDist.boundaryFieldRef()[patchi];
    for (Foam::label facei = 0; facei < yPatch.size(); ++facei)
    {
        yPatch[facei] = ofY[facei];
    }
}

Info << "Reading turbulence fields\n" << endl;
Foam::volScalarField ofNuTilda(
    Foam::IOobject(
        "nuTilda",
        runTime.timeName(),
        mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ),
    mesh
);

Foam::volScalarField ofNut(
    Foam::IOobject(
        "nut",
        runTime.timeName(),
        mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ),
    mesh
);

Foam::label pRefCell = 0;
Foam::scalar pRefValue = 0.0;
setRefCell(ofP, mesh.solutionDict().subDict("SIMPLE"), pRefCell, pRefValue);
// mesh.setFluxRequired(p.name());
