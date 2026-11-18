// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Compares NeoFOAM's atmospheric boundary layer BCs against OpenFOAM's atmosphericModels:
//
//   - atmBoundaryLayerInlet{Velocity,K,Epsilon}: an inletOutlet whose inlet value is the ABL
//     profile; the reader takes the profile from OpenFOAM's refValue(). `inlet` is an inflow
//     patch, and `outlet` carries an atmBoundaryLayerInletEpsilon on an outflow patch, as
//     atmospheric cases do, where it must act as zero gradient.
//   - atmNutkWallFunction / atmEpsilonWallFunction: the rough-wall functions with roughness
//     length z0 on `fixedWalls`. As for epsilonWallFunction, the face value is the BC's job and
//     the wall-adjacent cell pin and G feedback are KEpsilon::correct's, both at y + z0.
//
// setup_atmBoundaryLayer is setup_epsilonWallFunction with these BCs and the flow reversed so
// that `inlet` is an inflow.

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"

namespace fvc = Foam::fvc;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

TEST_CASE("atm boundary conditions: NeoFOAM boundary correction matches OpenFOAM")
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

    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );

    auto& ofK = mesh.lookupObjectRef<Foam::volScalarField>("k");
    auto& ofEps = mesh.lookupObjectRef<Foam::volScalarField>("epsilon");
    auto& ofNut = mesh.lookupObjectRef<Foam::volScalarField>("nut");

    const Foam::label wallId = mesh.boundaryMesh().findPatchID("fixedWalls");
    const Foam::label inletId = mesh.boundaryMesh().findPatchID("inlet");
    const Foam::label outletId = mesh.boundaryMesh().findPatchID("outlet");
    REQUIRE(wallId >= 0);
    REQUIRE(inletId >= 0);
    REQUIRE(outletId >= 0);
    REQUIRE(U.boundaryField()[inletId].type() == "atmBoundaryLayerInletVelocity");
    REQUIRE(ofK.boundaryField()[inletId].type() == "atmBoundaryLayerInletK");
    REQUIRE(ofEps.boundaryField()[inletId].type() == "atmBoundaryLayerInletEpsilon");
    REQUIRE(ofEps.boundaryField()[outletId].type() == "atmBoundaryLayerInletEpsilon");
    REQUIRE(ofEps.boundaryField()[wallId].type() == "atmEpsilonWallFunction");
    REQUIRE(ofNut.boundaryField()[wallId].type() == "atmNutkWallFunction");

    // The flow has to enter through `inlet` and leave through `outlet`, or the inletOutlet
    // switch is not exercised in both directions.
    REQUIRE(Foam::max(phi.boundaryField()[inletId]) < 0.0);
    REQUIRE(Foam::min(phi.boundaryField()[outletId]) > 0.0);

    // The cell-to-wall distance the wall functions consume as `y`.
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

    // The epsilon wall function's updateCoeffs() looks up the model's production field, which
    // kEpsilon only creates on its first correct(); same workaround as the epsilonWallFunction
    // test.
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

    SECTION("atm boundary conditions are registered with the volume factories")
    {
        const auto& sTable = fvcc::VolumeBoundaryFactory<NeoN::scalar>::table();
        REQUIRE(sTable.find("atmBoundaryLayerInlet") != sTable.end());
        REQUIRE(sTable.find("atmNutkWallFunction") != sTable.end());
        REQUIRE(sTable.find("atmEpsilonWallFunction") != sTable.end());
        const auto& vTable = fvcc::VolumeBoundaryFactory<NeoN::Vec3>::table();
        REQUIRE(vTable.find("atmBoundaryLayerInlet") != vTable.end());
    }

    SECTION("inlet profiles round-trip before any flux is seen (" + execName + ")")
    {
        // Without a flux every ABL face starts as inflow, holding the profile -- the value the
        // OpenFOAM constructors give the patch. constructFrom applies exactly that update.
        auto nfU = NeoFOAM::constructFrom(exec, nfMesh, U);
        auto nfK = NeoFOAM::constructFrom(exec, nfMesh, ofK);
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(U, ApproxVector(1e-14)));
        REQUIRE_THAT(nfK.boundaryData(), EqualsBoundary(ofK, ApproxScalar(1e-14)));
    }

    SECTION("face values agree after a flux-aware correction (" + execName + ")")
    {
        // Upstream: the inletOutlet switch on phi, and the rough-wall functions.
        U.correctBoundaryConditions();
        ofK.correctBoundaryConditions();
        ofEps.correctBoundaryConditions();
        ofNut.correctBoundaryConditions();

        auto nfU = NeoFOAM::constructFrom(exec, nfMesh, U);
        auto nfK = NeoFOAM::constructFrom(exec, nfMesh, ofK);
        auto nfEps = NeoFOAM::constructFrom(exec, nfMesh, ofEps);
        auto nfNut = NeoFOAM::constructFrom(exec, nfMesh, ofNut);
        auto nfNu = NeoFOAM::constructFrom(exec, nfMesh, nuFoam);
        auto nfNearWallDist = NeoFOAM::constructFrom(exec, nfMesh, nearWallDist);
        auto nfPhi = NeoFOAM::constructFrom(exec, nfMesh, phi);

        fvcc::BoundaryContext ctx;
        ctx.insert("k", nfK);
        ctx.insert("nu", nfNu);
        ctx.insert("nearWallDist", nfNearWallDist);
        ctx.insert("phi", nfPhi);
        nfU.correctBoundaryConditions(ctx);
        nfK.correctBoundaryConditions(ctx);
        nfEps.correctBoundaryConditions(ctx);
        nfNut.correctBoundaryConditions(ctx);

        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(U, ApproxVector(1e-12)));
        REQUIRE_THAT(nfK.boundaryData(), EqualsBoundary(ofK, ApproxScalar(1e-12)));
        // Covers the inlet profile, the outlet falling back to zero gradient and the rough-wall
        // epsilon C_mu^0.75 k^1.5/(kappa (y + z0)).
        REQUIRE_THAT(nfEps.boundaryData(), EqualsBoundary(ofEps, ApproxScalar(1e-12)));
        // nu (y+ kappa/log((y + z0)/z0) - 1).
        REQUIRE_THAT(nfNut.boundaryData(), EqualsBoundary(ofNut, ApproxScalar(1e-12)));
    }
}

// ============================================================
// The cell half of atmEpsilonWallFunction -- the wall-adjacent epsilon pin and the G feedback,
// both at y + z0 -- and the inletOutlet switch of k and epsilon inside KEpsilon::correct, against
// a full OpenFOAM kEpsilon::correct().
// ============================================================
TEST_CASE("atm boundary conditions: KEpsilon::correct matches OpenFOAM")
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
        == "atmEpsilonWallFunction"
    );

    const Foam::volScalarField wallDist = nf::makeNearWallDistField(mesh);

    auto& nfU = NeoFOAM::constructAndRegister(fieldCollection, rt, U, false);
    auto& nfP = NeoFOAM::constructAndRegister(fieldCollection, rt, p, false);
    auto& nfPhi = NeoFOAM::constructAndRegister(fieldCollection, rt, phi, false);
    auto& nfK = NeoFOAM::constructAndRegister(fieldCollection, rt, ofK, false);
    auto& nfEps = NeoFOAM::constructAndRegister(fieldCollection, rt, ofEps, false);

    auto [nfWallDist, nfNu, nfNut] =
        NeoFOAM::constFromMany(rt.exec, rt.nfMesh, wallDist, tnu(), ofNut);

    nf::KEpsilon turbNF(rt.exec, rt.nfMesh, nfNu, nfWallDist);
    // What KEpsilonModel does with the epsilon field it reads.
    turbNF.setWallRoughness(
        NeoN::Vector<NeoN::scalar>(rt.exec, nf::readEpsilonWallRoughness(ofEps, rt.nfMesh))
    );
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
    fvcc::BoundaryContext uCtx;
    uCtx.insert("phi", nfPhi);
    nfU.correctBoundaryConditions(uCtx);

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
    REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(U, ApproxVector(1e-10)));

    // Same bounds as the epsilonWallFunction test. The epsilon internal field is only right if
    // the wall-adjacent cells are pinned at y + z0; the boundary covers the rough-wall face
    // value and the inlet/outlet switch.
    REQUIRE_THAT(nfEps, EqualsInternal(epsFoam, ApproxScalar(1e-6)));
    REQUIRE_THAT(nfEps.boundaryData(), EqualsBoundary(epsFoam, ApproxScalar(1e-6)));

    REQUIRE_THAT(nfK, EqualsInternal(kFoam, ApproxScalar(1e-5)));
    REQUIRE_THAT(nfK.boundaryData(), EqualsBoundary(kFoam, ApproxScalar(1e-5)));

    REQUIRE_THAT(nfNut, EqualsInternal(nutFoam, ApproxScalar(1e-5)));
    REQUIRE_THAT(nfNut.boundaryData(), EqualsBoundary(nutFoam, ApproxScalar(1e-5)));
}
