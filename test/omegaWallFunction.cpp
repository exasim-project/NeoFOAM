// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Compares NeoFOAM's OmegaWallFunction patch BC against OpenFOAM's
// Foam::omegaWallFunctionFvPatchScalarField using the BINOMIAL (n=2) blender,
// which is upstream's default and the only blender NeoFOAM currently
// implements. Drives OF via the kOmegaSST turbulence model so the BC's
// updateCoeffs() runs against the same inputs (k, ν, y) we feed NeoFOAM.

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "wallDist.H"

using Catch::Approx;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

TEST_CASE("omegaWallFunction: NeoFOAM boundary correction matches OpenFOAM (BINOMIAL n=2)")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    auto nfMesh = mesh.nfMesh();

    // --- OpenFOAM side ---------------------------------------------------

    Foam::volVectorField U(
        Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ),
        mesh
    );

    Foam::surfaceScalarField phi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::READ_IF_PRESENT,
            Foam::IOobject::NO_WRITE
        ),
        Foam::fvc::flux(U)
    );

    // singlePhaseTransportModel exposes nu via the dict in constant/.
    Foam::singlePhaseTransportModel transport(U, phi);
    Foam::tmp<Foam::volScalarField> tnu = transport.nu();
    Foam::volScalarField nuFoam(
        Foam::IOobject(
            "nu",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tnu()
    );

    // Instantiating the turbulence model registers k, omega, nut from disk;
    // the registry is what omegaWallFunctionFvPatchScalarField::updateCoeffs()
    // walks via db().lookupObject<turbulenceModel>(...).
    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );

    auto& ofOmega = mesh.lookupObjectRef<Foam::volScalarField>("omega");
    const Foam::volScalarField& ofK = mesh.lookupObject<Foam::volScalarField>("k");

    const Foam::label wallId = mesh.boundaryMesh().findPatchID("fixedWalls");
    REQUIRE(wallId >= 0);
    REQUIRE(ofOmega.boundaryField()[wallId].type() == "omegaWallFunction");

    // Build a `nearWallDist` volScalarField whose boundary values are the
    // cell-to-wall-face distances exposed by turbulenceModel::y()[patchi] —
    // the same field NutUSpaldingWallFunction consumes via BoundaryContext.
    Foam::volScalarField nearWallDist(
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
    const auto& fbm = mesh.boundaryMesh();
    for (Foam::label patchi = 0; patchi < fbm.size(); ++patchi)
    {
        const Foam::scalarField& ofY = foamTurb->y()[patchi];
        Foam::fvPatchScalarField& yPatch = nearWallDist.boundaryFieldRef()[patchi];
        forAll(yPatch, facei)
        {
            yPatch[facei] = ofY[facei];
        }
    }

    // omegaWallFunctionFvPatchScalarField::updateCoeffs() looks up the
    // turbulence model's production field (`kOmegaSST:G`) in the registry —
    // that field is normally created lazily on the first kOmegaSST::correct()
    // call. We pre-register an empty one here so the BC update can run on a
    // standalone field read; we don't assert on G itself.
    Foam::volScalarField::Internal kOmegaSST_G(
        Foam::IOobject(
            foamTurb->GName(),
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::REGISTER
        ),
        mesh,
        Foam::dimensionedScalar(Foam::sqr(Foam::dimVelocity) / Foam::dimTime, Foam::Zero)
    );

    // Fire upstream's wall function. This is what writes the BINOMIAL-blended
    // ω onto the wall faces (and the adjacent wall cells, but we only assert
    // on the boundary face values).
    ofOmega.correctBoundaryConditions();

    SECTION("OmegaWallFunction is registered with the volume factory")
    {
        const auto& table = fvcc::VolumeBoundaryFactory<NeoN::scalar>::table();
        REQUIRE(table.find("omegaWallFunction") != table.end());
    }

    SECTION("boundary values agree after correction (" + execName + ")")
    {
        // Build the NeoFOAM-side fields. constructFrom dispatches the
        // `omegaWallFunction` patch entry through readers.hpp to our new
        // OmegaWallFunction class.
        auto nfOmega = NeoFOAM::constructFrom(exec, nfMesh, ofOmega);
        auto nfK = NeoFOAM::constructFrom(exec, nfMesh, ofK);
        auto nfNu = NeoFOAM::constructFrom(exec, nfMesh, nuFoam);
        auto nfNearWallDist = NeoFOAM::constructFrom(exec, nfMesh, nearWallDist);

        // Confirm internalField round-trip (also catches mesh ordering bugs
        // before we touch the BC).
        REQUIRE_THAT(nfK, EqualsInternal(ofK, ApproxScalar(1e-14)));

        // Drive our BC with the same inputs as OF.
        fvcc::BoundaryContext ctx;
        ctx.insert("k", nfK);
        ctx.insert("nu", nfNu);
        ctx.insert("nearWallDist", nfNearWallDist);
        nfOmega.correctBoundaryConditions(ctx);

        // OF's BINOMIAL(n=2) blender is ω = √(ωᵥᵢₛ² + ωₗₒg²) on each wall face;
        // both implementations compute it from the same coefficients
        // (β₁=0.075, C_µ=0.09, κ=0.41) and the same (k, ν, y) inputs, so the
        // agreement should be near floating-point exact.
        REQUIRE_THAT(nfOmega.boundaryData(), EqualsBoundary(ofOmega, ApproxScalar(1e-12)));
    }
}
