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

    SECTION("rAU" + execName)
    {
        nf::compare(nfU, ofU, ApproxVector(epsilon));
        nf::compare(nfNu, ofNu, ApproxScalar(epsilon));
        nf::compare(nfPhi, ofPhi, ApproxScalar(epsilon));

        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        nfUEqn.assemble();

        nf::compare(
            NeoN::la::upper(nfUEqn.linearSystem().matrix()),
            ofUEqn.upper(),
            ApproxVector(1e-15)
        );
        nf::compare(
            NeoN::la::removeBoundaryContributions(
                nfUEqn.linearSystem(),
                NeoN::Vector<NeoN::Vec3>(exec, 0)  // serial: no proc-faces
            ).matrix().diag(),
            ofUEqn.diag(),
            ApproxVector(1e-15)
        );

        auto nfrAU = nf::computeRAU(nfUEqn);

        NeoFOAM::compare(nfrAU, forAU, ApproxScalar(1e-15), true);
    }

    SECTION("HbyA" + execName)
    {
        nf::compare(nfU, ofU, ApproxVector(epsilon));
        nf::compare(nfPhi, ofPhi, ApproxScalar(epsilon));
        nf::compare(nfUEqn.linearSystem().rhs(), ofUEqn.source(), ApproxVector(epsilon), false);

        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());

        nfUEqn.assemble();
        auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);

        nf::compare(nfHbyA, HbyA, ApproxVector({1e-08, 1e-08, 1e-02}));
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
        nf::compare(nfHbyA, ofConstrainHbyA, ApproxVector({1e-08, 1e-08, 1e-02}));
    }

    SECTION("compute flux")
    {
        nf::compare(nfPhi, ofPhi, ApproxScalar(epsilon));
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

        nf::compare(pEqn.linearSystem().matrix().diag(), ofpEqn.diag(), ApproxScalar(1e-15));
        nf::compare(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            ofpEqn.upper(),
            ApproxScalar(1e-15)
        );

        nf::updateFaceVelocity(nfPhi, pEqn, nfPhi0);
        nf::compare(nfPhi0, ofPhi0, ApproxScalar(1e-15));
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

        Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));
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
}
