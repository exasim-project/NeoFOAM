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

Foam::wallDist y(mesh);

Info << "Reading turbulence fields\n" << endl;
Foam::volScalarField
    ofK(Foam::IOobject(
            "k",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        mesh);

// Read the active RAS model name from constant/turbulenceProperties so we know
// which secondary scalar field to expect on disk (omega for kOmegaSST,
// epsilon for kEpsilon, etc.). The whole loop below branches on this string.
const Foam::word rasModelName = Foam::IOdictionary(Foam::IOobject(
                                                       "turbulenceProperties",
                                                       runTime.constant(),
                                                       mesh,
                                                       Foam::IOobject::MUST_READ_IF_MODIFIED,
                                                       Foam::IOobject::NO_WRITE,
                                                       Foam::IOobject::NO_REGISTER
                                                   ))
                                    .subDict("RAS")
                                    .get<Foam::word>("RASModel");

Info << "RAS model: " << rasModelName << endl;

const bool isKOmegaSST = (rasModelName == "kOmegaSST");
const bool isKEpsilon = (rasModelName == "kEpsilon");
if (!isKOmegaSST && !isKEpsilon)
{
    FatalErrorInFunction << "Unsupported RAS model '" << rasModelName << "'. "
                         << "neoSimpleFoam currently supports: kOmegaSST, kEpsilon."
                         << Foam::abort(Foam::FatalError);
}

// omega and epsilon are both attempted with READ_IF_PRESENT so the file that
// matches the chosen model is read; the other is created as a placeholder
// (won't be touched by the active turbulence model).
Foam::volScalarField ofOmega(
    Foam::IOobject(
        "omega",
        runTime.timeName(),
        mesh,
        isKOmegaSST ? Foam::IOobject::MUST_READ : Foam::IOobject::READ_IF_PRESENT,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ),
    mesh,
    Foam::dimensionedScalar("omega", Foam::dimless / Foam::dimTime, 1.0)
);

Foam::volScalarField ofEpsilon(
    Foam::IOobject(
        "epsilon",
        runTime.timeName(),
        mesh,
        isKEpsilon ? Foam::IOobject::MUST_READ : Foam::IOobject::READ_IF_PRESENT,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ),
    mesh,
    Foam::dimensionedScalar(
        "epsilon",
        Foam::dimLength* Foam::dimLength / (Foam::dimTime * Foam::dimTime * Foam::dimTime),
        1.0
    )
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
