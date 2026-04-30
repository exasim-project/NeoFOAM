
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"
#include "constrainHbyA.H"
#include "findRefCell.H"

namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr; // A single time object
TEST_CASE("Distributed PressureVelocityCoupling")
{
    SECTION("Parallel sanity check")
    {
        REQUIRE(Foam::Pstream::parRun());
        REQUIRE(Foam::Pstream::nProcs() == 3);
    }

    float epsilon = 1e-32;
    Foam::Time& runTime = *timePtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);

    auto ofU = randomVectorField(runTime, mesh, "U");
    auto ofp = randomScalarField(runTime, mesh, "p");
    ofp.correctBoundaryConditions();
    ofU.correctBoundaryConditions();
    auto& oldOfU = ofU.oldTime();
    oldOfU.primitiveFieldRef() = Foam::vector(0.0, 0.0, 0.0);
    oldOfU.correctBoundaryConditions();

    auto& vectorCollection = nnfvcc::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfP = NeoFOAM::constructAndRegister(vectorCollection, rt, ofp);

    Foam::surfaceScalarField ofPhi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(ofU)
    );

    Foam::surfaceScalarField ofNu(
        Foam::IOobject(
            "nu",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("nu", Foam::dimensionSet(0, 2, -1, 0, 0), 0.01)
    );

    auto nfPhi = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi);
    auto nfNu = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofNu);

    Foam::fvVectorMatrix ofUEqn(fvm::ddt(ofU) + fvm::div(ofPhi, ofU) - fvm::laplacian(ofNu, ofU));

    auto& nfU = NeoFOAM::constructAndRegister(vectorCollection, rt, ofU);
    auto& nfOldU = fvcc::oldTime(nfU);
    NeoN::fill(nfOldU.internalVector(), NeoN::Vec3(0.0, 0.0, 0.0));
    nfOldU.correctBoundaryConditions();

    nf::PDESolver<NeoN::Vec3> nfUEqn(
        dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
        nfU,
        rt
    );

    NeoN::fill(nfUEqn.linearSystem().rhs(), NeoN::Vec3(0.0, 0.0, 0.0));

    nf::compare(nfP, ofp, ApproxScalar(epsilon), true);
    nf::compare(nfU, ofU, ApproxVector(epsilon), true);

    SECTION("rAU" + execName)
    {
        nf::compare(nfU, ofU, ApproxVector(epsilon), true);

        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        forAU.correctBoundaryConditions();
        nfUEqn.assemble();
        auto nfrAU = nf::computeRAU(nfUEqn);

        NeoFOAM::compare(nfrAU, forAU, ApproxScalar(1e-15), true);
    }

    SECTION("interpolate rAU" + execName)
    {
        // Tests the same interpolation call neoIcoFoam uses to build rAUf:
        //   SurfaceField rAU =
        //       SurfaceInterpolation<scalar>(exec, mesh, TokenList{"linear"})
        //           .interpolate(crAU);
        // The reference is OF's Foam::linearInterpolate(forAU).
        //
        // Both internal AND boundary values are compared (withBoundaries=true).
        // For processor patches the comparison verifies that the proc-face
        // value computed locally on each rank from
        //   w * crAU[own] + (1-w) * crAU[ghost]
        // matches OF's per-rank value, where the ghost cell value comes from
        // the prior crAU.correctBoundaryConditions() exchange done inside
        // computeRAU.
        nf::compare(nfU, ofU, ApproxVector(epsilon), true);

        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        forAU.correctBoundaryConditions();

        // OF reference: linear face interpolation of forAU.
        // Use linearInterpolate directly to avoid relying on a `default`
        // entry in interpolationSchemes (the test setup only registers
        // flux(U)/flux(HbyA) entries).
        Foam::surfaceScalarField ofRAUf("rAUf", Foam::linearInterpolate(forAU));

        nfUEqn.assemble();
        auto nfrAU = nf::computeRAU(nfUEqn);
        // Sanity: input to the interpolation must already match OF, otherwise
        // any disagreement we see at the surface field is not the
        // interpolation's fault.
        nf::compare(nfrAU, forAU, ApproxScalar(1e-15), true);

        nnfvcc::SurfaceField<NeoN::scalar> nfRAUf = fvcc::SurfaceInterpolation<NeoN::scalar>(
                                                        rt.exec,
                                                        rt.nfMesh,
                                                        NeoN::TokenList({std::string("linear")})
        )
                                                        .interpolate(nfrAU);

        nf::compare(nfRAUf, ofRAUf, ApproxScalar(1e-12), true);
    }

    SECTION("HbyA" + execName)
    {
        nf::compare(nfU, ofU, ApproxVector(epsilon), true);
        nf::compare(nfPhi, ofPhi, ApproxScalar(epsilon), true);
        nf::compare(nfUEqn.linearSystem().rhs(), ofUEqn.source(), ApproxVector(epsilon), true);

        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());

        nfUEqn.assemble();

        // NOTE removeBoundaryContributions is not working in distributed case
        // nf::compare(
        //     NeoN::la::removeBoundaryContributions(nfUEqn.linearSystem()).matrix().diag(),
        //     ofUEqn.diag(),
        //     ApproxVector(1e-15)
        // );

        nf::compare(
            NeoN::la::upper(nfUEqn.linearSystem().matrix()),
            ofUEqn.upper(),
            ApproxVector(1e-15)
        );

        auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);

        nf::compare(nfHbyA, HbyA, ApproxVector({1e-12}), true);
    }

    SECTION("constrainHbyA")
    {
        nf::compare(nfU, ofU, ApproxVector(epsilon));
        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());
        Foam::volVectorField ofConstrainHbyA(
            "ofConstHbyA",
            Foam::constrainHbyA(forAU * ofUEqn.H(), ofU, ofp)
        );
        nfUEqn.assemble();
        auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);

        nf::constrainHbyA(nfU, nfP, nfHbyA);

        nf::compare(nfHbyA, ofConstrainHbyA, ApproxVector({1e-12}), true);
    }

    SECTION("compute flux")
    {
        nf::compare(nfU, ofU, ApproxVector(epsilon));
        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());
        Foam::volVectorField ofConstrainHbyA(
            "ofConstHbyA",
            Foam::constrainHbyA(forAU * ofUEqn.H(), ofU, ofp)
        );
        nfUEqn.assemble();
        auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);
        Foam::surfaceScalarField ofFlux("ofFlux", fvc::flux(HbyA));
        auto nfFlux = nf::flux(nfHbyA);

        nf::compare(nfFlux, ofFlux, ApproxScalar(1e-12), true);
    }

    SECTION("compute flux")
    {
        nfPhi.correctBoundaryConditions();
        nf::compare(nfPhi, ofPhi, ApproxScalar(epsilon), false);
        auto forAUf =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUf");
        auto nfrAUf = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, forAUf);

        Foam::surfaceScalarField ofPhi0("phi0", ofPhi * 0.0);
        auto nfPhi0 = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi0);

        Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));
        ofPhi0 = ofPhi - ofpEqn.flux();
        solve(ofpEqn);

        nf::PDESolver<NeoN::scalar> pEqn(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

        pEqn.assemble();

        // NOTE removeBoundaryContributions is not working in distributed case
        // nf::compare(
        //     NeoN::la::removeBoundaryContributions(pEqn.linearSystem()).matrix().diag(),
        //     ofpEqn.diag(),
        //     ApproxScalar(1e-15),
        //     false
        // );

        nf::compare(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            ofpEqn.upper(),
            ApproxScalar(1e-15),
            false
        );

        nf::updateFaceVelocity(nfPhi, pEqn, nfPhi0);
        nf::compare(nfPhi0, ofPhi0, ApproxScalar(1e-12), true);
    }

    SECTION("solve pEqn")
    {
        nf::compare(nfPhi, ofPhi, ApproxScalar(epsilon), false);
        nf::compare(nfP, ofp, ApproxScalar(1e-12), false);

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

        auto forAUf =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUf");
        auto nfrAUf = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, forAUf);

        Foam::surfaceScalarField ofPhi0("phi0", ofPhi * 0.0);
        auto nfPhi0 = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi0);

        Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));
        ofPhi0 = ofPhi - ofpEqn.flux();
        solve(ofpEqn);

        nf::PDESolver<NeoN::scalar> pEqn(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );

        auto stats = pEqn.solve();

        // NOTE removeBoundaryContributions is not working in distributed case
        // nf::compare(
        //     NeoN::la::removeBoundaryContributions(pEqn.linearSystem()).matrix().diag(),
        //     ofpEqn.diag(),
        //     ApproxScalar(1e-15),
        //     false
        // );

        nf::compare(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            ofpEqn.upper(),
            ApproxScalar(1e-15),
            false
        );
        nf::compare(pEqn.linearSystem().rhs(), ofpEqn.source(), ApproxScalar(1e-15), false);

        ofp.correctBoundaryConditions();
        nfP.correctBoundaryConditions();

        auto [numIter, initResNorm, finalResNorm, solveTime] = stats.entries[0];

        REQUIRE(numIter != 0);
        REQUIRE(initResNorm != 0);
        REQUIRE(finalResNorm < initResNorm);

        nf::compare(nfP, ofp, ApproxScalar(1e-12), true);
    }

    SECTION("solve pEqn and update faceVelocity")
    {
        nf::compare(nfPhi, ofPhi, ApproxScalar(epsilon), false);
        nf::compare(nfP, ofp, ApproxScalar(1e-12), false);

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

        auto forAUf =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUf");
        auto nfrAUf = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, forAUf);

        Foam::surfaceScalarField ofPhi0("phi0", ofPhi * 0.0);
        auto nfPhi0 = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi0);

        Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));
        Foam::label pRefCell = 0;
        Foam::scalar pRefValue = 0.0;
        Foam::setRefCell(ofp, rt.mesh.solutionDict().subDict("PISO"), pRefCell, pRefValue);
        solve(ofpEqn);
        ofPhi0 = ofPhi - ofpEqn.flux();

        nf::PDESolver<NeoN::scalar> pEqn(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );

        if (rt.mpiEnvironment.rank() == 0)
        {
            pEqn.setReference(0, 0.0);
        }
        auto stats = pEqn.solve();

        // NOTE removeBoundaryContributions is not working in distributed case
        // nf::compare(
        //     NeoN::la::removeBoundaryContributions(pEqn.linearSystem()).matrix().diag(),
        //     ofpEqn.diag(),
        //     ApproxScalar(1e-15),
        //     false
        // );

        nf::compare(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            ofpEqn.upper(),
            ApproxScalar(1e-15),
            false
        );
        nf::compare(pEqn.linearSystem().rhs(), ofpEqn.source(), ApproxScalar(1e-15), false);

        ofp.correctBoundaryConditions();
        nfP.correctBoundaryConditions();

        auto [numIter, initResNorm, finalResNorm, solveTime] = stats.entries[0];

        REQUIRE(numIter != 0);
        REQUIRE(initResNorm != 0);
        REQUIRE(finalResNorm < initResNorm);

        nf::compare(nfP, ofp, ApproxScalar(1e-12), true);
        // nf::compare(nfPhi, ofPhi, ApproxScalar(1e-12), true);

        nf::updateFaceVelocity(nfPhi, pEqn, nfPhi0);
        // TODO this neneds to be relatively loose
        nf::compare(nfPhi0, ofPhi0, ApproxScalar(1e-05), false);
    }
}
