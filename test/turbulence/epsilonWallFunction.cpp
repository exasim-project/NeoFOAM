// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Compares NeoFOAM's epsilonWallFunction against
// Foam::epsilonWallFunctionFvPatchScalarField, which does two separate things:
//
//   1. it writes the blended dissipation onto the wall FACE, and
//   2. via manipulateMatrix/fvMatrix::setValues it fixes epsilon in the
//      wall-adjacent CELLS instead of solving for them.
//
// NeoFOAM splits those halves: the BC owns (1), KEpsilon::correct owns (2) by
// constraining the epsilon equation. Both are covered here -- the first
// section drives the BC alone, the second runs a full correct() against
// OpenFOAM's kEpsilon so the cell pin is exercised too.
//
// The sibling kEpsilon test runs on setup_kEpsilon, whose epsilon is a plain
// fixedValue at the walls, so neither half is reachable there. This test uses
// setup_epsilonWallFunction, which differs only in carrying
// epsilonWallFunction/nutkWallFunction on `fixedWalls`.

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "wallDist.H"

using Catch::Approx;

namespace fvc = Foam::fvc;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

TEST_CASE("epsilonWallFunction: NeoFOAM boundary correction matches OpenFOAM")
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
        fvc::flux(U)
    );

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

    // Instantiating the model registers k, epsilon, nut; the registry is what
    // epsilonWallFunctionFvPatchScalarField::updateCoeffs() walks.
    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );

    auto& ofEps = mesh.lookupObjectRef<Foam::volScalarField>("epsilon");
    const Foam::volScalarField& ofK = mesh.lookupObject<Foam::volScalarField>("k");

    const Foam::label wallId = mesh.boundaryMesh().findPatchID("fixedWalls");
    REQUIRE(wallId >= 0);
    REQUIRE(ofEps.boundaryField()[wallId].type() == "epsilonWallFunction");

    // nearWallDist boundary values are the cell-to-wall distances the BC
    // consumes as `y`, taken from turbulenceModel::y() exactly as the
    // omegaWallFunction test does.
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

    // epsilonWallFunctionFvPatchScalarField::updateCoeffs() looks up the
    // model's production field (`kEpsilon:G`), which is normally created
    // lazily on the first kEpsilon::correct(). Pre-register an empty one so
    // the BC update can run standalone; we do not assert on G itself. Same
    // workaround as the omegaWallFunction test.
    Foam::volScalarField::Internal kEpsilonG(
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

    // Fire upstream's wall function so the face values are the blended ones.
    ofEps.correctBoundaryConditions();

    SECTION("EpsilonWallFunction is registered with the volume factory")
    {
        const auto& table = fvcc::VolumeBoundaryFactory<NeoN::scalar>::table();
        REQUIRE(table.find("epsilonWallFunction") != table.end());
    }

    SECTION("wall face values agree after correction (" + execName + ")")
    {
        auto nfEps = NeoFOAM::constructFrom(exec, nfMesh, ofEps);
        auto nfK = NeoFOAM::constructFrom(exec, nfMesh, ofK);
        auto nfNu = NeoFOAM::constructFrom(exec, nfMesh, nuFoam);
        auto nfNearWallDist = NeoFOAM::constructFrom(exec, nfMesh, nearWallDist);

        // Round-trip first, so a mesh-ordering bug does not masquerade as a
        // wall-function mismatch below.
        REQUIRE_THAT(nfK, EqualsInternal(ofK, ApproxScalar(1e-14)));

        fvcc::BoundaryContext ctx;
        ctx.insert("k", nfK);
        ctx.insert("nu", nfNu);
        ctx.insert("nearWallDist", nfNearWallDist);
        nfEps.correctBoundaryConditions(ctx);

        // Both sides evaluate the BINOMIAL n=2 blend sqrt(eVis^2 + eLog^2)
        // from the same (k, nu, y) and the same coefficients, so this should
        // be near floating-point exact.
        REQUIRE_THAT(nfEps.boundaryData(), EqualsBoundary(ofEps, ApproxScalar(1e-12)));
    }
}

// ============================================================
// The cell half: OpenFOAM fixes epsilon in the wall-adjacent cells rather than
// solving for them, so a correct() that only sets the face value leaves those
// cells under-dissipated and k over-predicted. Mirrors the wrapper comparison
// in kEpsilon.cpp, but on a fixture that actually carries the wall function.
// ============================================================
TEST_CASE("epsilonWallFunction: near-wall cells are pinned as OpenFOAM does")
{
    Foam::Time& runTime = *timePtr;

    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
    solverDict.subDict("k") = nf::mapFvSolution(solverDict.subDict("k"));
    solverDict.subDict("epsilon") = nf::mapFvSolution(solverDict.subDict("epsilon"));
    rt.fvSchemesDict = nf::mapFvSchemes(rt.fvSchemesDict);
    const NeoN::Dictionary controlDict = nf::convert(runTime.controlDict());
    runTime.setDeltaT(controlDict.get<Foam::scalar>("deltaT"));

    Foam::volVectorField U(
        Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ),
        mesh
    );
    Foam::volScalarField p(
        Foam::IOobject("p", runTime.timeName(), mesh, Foam::IOobject::MUST_READ),
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
        fvc::flux(U)
    );

    Foam::singlePhaseTransportModel transport(U, phi);
    auto tnu = transport.nu();

    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );
    foamTurb->validate();

    const Foam::volScalarField& ofK = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::volScalarField& ofEps = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const Foam::volScalarField& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");

    REQUIRE(
        ofEps.boundaryField()[mesh.boundaryMesh().findPatchID("fixedWalls")].type()
        == "epsilonWallFunction"
    );

    // Build it the way production does: the per-patch near-wall distance, on the
    // boundary. Foam::wallDist would leave the wall faces at ~0 and blow up eLog ~ 1/y.
    const Foam::volScalarField wallDist = nf::makeNearWallDistField(mesh);

    auto& nfU = NeoFOAM::constructAndRegister(fieldCollection, rt, U, false);
    auto& nfP = NeoFOAM::constructAndRegister(fieldCollection, rt, p, false);
    auto& nfPhi = NeoFOAM::constructAndRegister(fieldCollection, rt, phi, false);
    auto& nfK = NeoFOAM::constructAndRegister(fieldCollection, rt, ofK, false);
    auto& nfEps = NeoFOAM::constructAndRegister(fieldCollection, rt, ofEps, false);

    auto [nfWallDist, nfNu, nfNut] =
        NeoFOAM::constFromMany(rt.exec, rt.nfMesh, wallDist, tnu(), ofNut);

    nf::KEpsilon turbNF(rt.exec, rt.nfMesh, nfNu, nfWallDist);
    turbNF.validate(nfU, nfK, nfEps, nfNut);

    fvcc::rotateOldTimes(nfU);
    fvcc::rotateOldTimes(nfPhi);
    fvcc::rotateOldTimes(nfK);
    fvcc::rotateOldTimes(nfEps);

    nf::PDESolver<NeoN::Vec3> UEqn(
        dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(turbNF.nuEff(), nfU)
            + dsl::exp::viscousStress(nfNu, nfNut, turbNF.gradU()),
        nfU,
        rt
    );
    UEqn.solve(-1.0 * dsl::exp::grad(nfP));
    nfU.correctBoundaryConditions();

    turbNF.correct(nfU, nfPhi, nfK, nfEps, nfNut, rt);

    // --- OpenFOAM reference ---
    Foam::fvVectorMatrix ofUEqn(
        Foam::fvm::ddt(U) + Foam::fvm::div(phi, U) + foamTurb->divDevReff(U)
    );
    Foam::solve(ofUEqn == -fvc::grad(p));
    U.correctBoundaryConditions();

    foamTurb->correct();

    const Foam::volScalarField& kFoam = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::volScalarField& epsFoam = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const Foam::volScalarField& nutFoam = mesh.lookupObject<Foam::volScalarField>("nut");

    REQUIRE_THAT(nfU, EqualsInternal(U, ApproxVector(1e-10)));

    // epsilon lands within 1e-8 of upstream, so it is held far tighter than the sibling
    // kEpsilon test's 5e-4. k keeps a looser bound: OpenFOAM's kEpsilon carries a
    // -(2/3)div(U) compressibility correction that is non-zero for a single uncoupled step,
    // worth ~2e-6 here. nut = Cmu k^2/eps inherits roughly twice k's relative error, hence
    // 1e-5 on a field of O(1e-3).
    //
    // The epsilon internal-field check is the point of this test: with the wall function
    // present, the wall-adjacent cells are only right if the epsilon equation pins them
    // the way fvMatrix::setValues does upstream. Disabling the pin in KEpsilon::correct
    // fails this assertion and this assertion alone.
    REQUIRE_THAT(nfEps, EqualsInternal(epsFoam, ApproxScalar(1e-6)));
    REQUIRE_THAT(nfEps.boundaryData(), EqualsBoundary(epsFoam, ApproxScalar(1e-6)));

    REQUIRE_THAT(nfK, EqualsInternal(kFoam, ApproxScalar(1e-5)));
    REQUIRE_THAT(nfK.boundaryData(), EqualsBoundary(kFoam, ApproxScalar(1e-5)));

    REQUIRE_THAT(nfNut, EqualsInternal(nutFoam, ApproxScalar(1e-5)));
}
