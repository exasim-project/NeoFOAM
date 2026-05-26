// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"
#include "constrainHbyA.H"

namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;
namespace dsl = NeoN::dsl;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using NeoFOAM::EqualsBoundary;
using NeoFOAM::EqualsInternal;

extern Foam::Time* timePtr; // A single time object


TEST_CASE("PressureVelocityCoupling")
{
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
    ofPhi.correctBoundaryConditions();

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
    ofNu.correctBoundaryConditions();

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

    SECTION("rAU" + execName)
    {
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector(epsilon)));

        REQUIRE_THAT(nfNu, EqualsInternal(ofNu, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfNu.boundaryData(), EqualsBoundary(ofNu, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfPhi, EqualsInternal(ofPhi, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfPhi.boundaryData(), EqualsBoundary(ofPhi, ApproxScalar(epsilon)));

        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        nfUEqn.assemble();

        REQUIRE_THAT(
            NeoN::la::upper(nfUEqn.linearSystem().matrix()),
            EqualsInternal(ofUEqn.upper(), ApproxVector(1e-15))
        );

        // NeoN stores boundary diagonal contributions directly in the matrix, whereas
        // OpenFOAM keeps them separate. Remove them before comparing against OpenFOAM
        // coefficients.
        REQUIRE_THAT(
            NeoN::la::removeBoundaryContributions(nfUEqn.linearSystem()).matrix().diag(),
            EqualsInternal(ofUEqn.diag(), ApproxVector(1e-15))
        );

        auto nfrAU = nf::computeRAU(nfUEqn);

        REQUIRE_THAT(nfrAU, EqualsInternal(forAU, ApproxScalar(1e-15)));
        REQUIRE_THAT(nfrAU.boundaryData(), EqualsBoundary(forAU, ApproxScalar(1e-15)));
    }

    SECTION("HbyA" + execName)
    {
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfPhi, EqualsInternal(ofPhi, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfPhi.boundaryData(), EqualsBoundary(ofPhi, ApproxScalar(epsilon)));
        REQUIRE_THAT(
            nfUEqn.linearSystem().rhs(),
            EqualsInternal(ofUEqn.source(), ApproxVector(epsilon))
        );

        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());

        nfUEqn.assemble();
        auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);

        REQUIRE_THAT(nfHbyA, EqualsInternal(HbyA, ApproxVector({1e-08, 1e-08, 1e-02})));
        REQUIRE_THAT(
            nfHbyA.boundaryData(),
            EqualsBoundary(HbyA, ApproxVector({1e-08, 1e-08, 1e-02}))
        );
    }

    SECTION("constrainHbyA")
    {
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector(epsilon)));
        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());
        Foam::volVectorField ofConstrainHbyA(
            "ofConstHbyA",
            Foam::constrainHbyA(forAU * ofUEqn.H(), ofU, ofp)
        );
        nfUEqn.assemble();
        auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);

        nf::constrainHbyA(nfU, nfP, nfHbyA);
        REQUIRE_THAT(nfHbyA, EqualsInternal(ofConstrainHbyA, ApproxVector({1e-08, 1e-08, 1e-02})));
        REQUIRE_THAT(
            nfHbyA.boundaryData(),
            EqualsBoundary(ofConstrainHbyA, ApproxVector({1e-08, 1e-08, 1e-02}))
        );
    }

    SECTION("compute flux")
    {
        REQUIRE_THAT(nfPhi, EqualsInternal(ofPhi, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfPhi.boundaryData(), EqualsBoundary(ofPhi, ApproxScalar(epsilon)));
        auto forAUf =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUf");
        auto nfrAUf = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, forAUf);

        Foam::surfaceScalarField ofPhi0("phi0", ofPhi * 0.0);
        auto nfPhi0 = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi0);

        Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));
        ofPhi0 = ofPhi - ofpEqn.flux();

        nf::PDESolver<NeoN::scalar> pEqn(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

        pEqn.assemble();

        REQUIRE_THAT(
            pEqn.linearSystem().matrix().diag(),
            EqualsInternal(ofpEqn.diag(), ApproxScalar(1e-15))
        );
        REQUIRE_THAT(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            EqualsInternal(ofpEqn.upper(), ApproxScalar(1e-15))
        );

        nf::updateFaceVelocity(nfPhi, pEqn, nfPhi0);
        REQUIRE_THAT(nfPhi0, EqualsInternal(ofPhi0, ApproxScalar(1e-15)));
        REQUIRE_THAT(nfPhi0.boundaryData(), EqualsBoundary(ofPhi0, ApproxScalar(1e-15)));
    }

    SECTION("assemble pEqn")
    {
        REQUIRE_THAT(nfPhi, EqualsInternal(ofPhi, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfP, EqualsInternal(ofp, ApproxScalar(1e-12)));

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

        auto forAUf =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUf");
        auto nfrAUf = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, forAUf);

        Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));
        solve(ofpEqn);


        nf::PDESolver<NeoN::scalar> pEqn(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );

        auto stats = pEqn.assemble();

        REQUIRE_THAT(
            NeoN::la::removeBoundaryContributions(pEqn.linearSystem()).matrix().diag(),
            EqualsInternal(ofpEqn.diag(), ApproxScalar(1e-15))
        );

        REQUIRE_THAT(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            EqualsInternal(ofpEqn.upper(), ApproxScalar(1e-15))
        );
    }

    SECTION("solve pEqn")
    {
        REQUIRE_THAT(nfPhi, EqualsInternal(ofPhi, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfP, EqualsInternal(ofp, ApproxScalar(1e-12)));

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

        auto forAUf =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUf");
        auto nfrAUf = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, forAUf);

        Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));
        solve(ofpEqn);


        nf::PDESolver<NeoN::scalar> pEqn(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );

        auto stats = pEqn.solve();

        REQUIRE_THAT(
            NeoN::la::removeBoundaryContributions(pEqn.linearSystem()).matrix().diag(),
            EqualsInternal(ofpEqn.diag(), ApproxScalar(1e-15))
        );

        REQUIRE_THAT(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            EqualsInternal(ofpEqn.upper(), ApproxScalar(1e-15))
        );

        REQUIRE_THAT(
            pEqn.linearSystem().rhs(),
            EqualsInternal(ofpEqn.source(), ApproxScalar(1e-15))
        );

        ofp.correctBoundaryConditions();
        nfP.correctBoundaryConditions();

        auto [numIter, initResNorm, finalResNorm, solveTime] = stats.entries[0];

        REQUIRE(numIter != 0);
        REQUIRE(initResNorm != 0);
        REQUIRE(finalResNorm < initResNorm);
        REQUIRE_THAT(nfP, EqualsInternal(ofp, ApproxScalar(1e-12)));
        REQUIRE_THAT(nfP.boundaryData(), EqualsBoundary(ofp, ApproxScalar(1e-12)));
    }
}
