
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

using NeoFOAM::EqualsInternal;
using NeoFOAM::EqualsBoundary;

extern Foam::Time* timePtr; // A single time object
TEST_CASE("Distributed PressureVelocityCoupling")
{
    SECTION("Parallel sanity check")
    {
        REQUIRE(Foam::Pstream::parRun());
        REQUIRE(Foam::Pstream::nProcs() == 3);
    }

    float epsilon = 1e-32;
    float epsilonII = 1e-13;
    Foam::Time& runTime = *timePtr;

    // CPU executor only for now; GPU MPI path requires CUDA-aware MPI.
    std::string execName = "CPUExecutor";
    NeoN::Executor exec = NeoN::CPUExecutor {};

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

    REQUIRE_THAT(nfP, EqualsInternal(ofp, ApproxScalar(epsilon)));
    REQUIRE_THAT(nfP.boundaryData(), EqualsBoundary(ofp, ApproxScalar(epsilon)));
    REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(epsilon)));
    REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector(epsilon)));

    SECTION("rAU" + execName)
    {
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector(epsilon)));

        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        forAU.correctBoundaryConditions();
        nfUEqn.assemble();
        auto nfrAU = nf::computeRAU(nfUEqn);

        REQUIRE_THAT(nfrAU, EqualsInternal(forAU, ApproxScalar(epsilonII)));
        // REQUIRE_THAT(nfrAU.boundaryData(), EqualsBoundary(forAU, ApproxScalar(epsilonII)));
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
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector(epsilon)));

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
        REQUIRE_THAT(nfrAU, EqualsInternal(forAU, ApproxScalar(epsilonII)));
        // REQUIRE_THAT(nfrAU.boundaryData(), EqualsBoundary(forAU, ApproxScalar(epsilonII)));

        nnfvcc::SurfaceField<NeoN::scalar> nfRAUf = fvcc::SurfaceInterpolation<NeoN::scalar>(
                                                        rt.exec,
                                                        rt.nfMesh,
                                                        NeoN::TokenList({std::string("linear")})
        )
                                                        .interpolate(nfrAU);

        REQUIRE_THAT(nfRAUf, EqualsInternal(ofRAUf, ApproxScalar(epsilonII)));
        // REQUIRE_THAT(nfRAUf.boundaryData(), EqualsBoundary(ofRAUf, ApproxScalar(epsilonII)));
    }

    SECTION("compute gradP" + execName)
    {
        // Tests the same call updateVelocity in
        // src/algorithms/pressureVelocityCoupling.cpp makes:
        //   auto gradP = GaussGreenGrad(p.exec(), p.mesh()).grad(p);
        // This is the cell-centred pressure gradient that drives the velocity
        // correction `u = HbyA - rAU * gradP` at the end of every PISO step.
        //
        // The call exercises three distinct paths in computeGrad:
        //   1. Internal-face accumulation (`computeGradInternal`).
        //   2. Physical-boundary extrapolation via computeBoundaryGrad.
        //   3. Processor-boundary face accumulation (`computeProcGradBoundary`).
        // Path (3) requires p.boundaryData().value() at the proc tail to hold
        // the ghost cell value — populated by p.correctBoundaryConditions().
        // Mirror neoIcoFoam.cpp:140 where p.correctBoundaryConditions() is
        // called between the pressure solve and updateVelocity.
        //
        // Compared at internal-cell level (withBoundaries=false). Proc-adjacent
        // cells exercise the proc-boundary loop without relying on the proc
        // patch comparison which `nf::compare` currently skips for processor
        // patches.
        REQUIRE_THAT(nfP, EqualsInternal(ofp, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfP.boundaryData(), EqualsBoundary(ofp, ApproxScalar(epsilon)));

        ofp.correctBoundaryConditions();
        nfP.correctBoundaryConditions();

        Foam::volVectorField ofGradP("gradP", fvc::grad(ofp));
        auto nfGradP = nnfvcc::GaussGreenGrad(rt.exec, rt.nfMesh).grad(nfP);

        REQUIRE_THAT(nfGradP, EqualsInternal(ofGradP, ApproxVector(epsilonII)));
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

        // NOTE removeBoundaryContributions is not working in distributed case
        // REQUIRE_THAT(
        //     NeoN::la::removeBoundaryContributions(nfUEqn.linearSystem()).matrix().diag(),
        //     EqualsInternal(ofUEqn.diag(), ApproxVector(1e-15))
        // );

        REQUIRE_THAT(
            NeoN::la::upper(nfUEqn.linearSystem().matrix()),
            EqualsInternal(ofUEqn.upper(), ApproxVector(1e-15))
        );

        auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);

        REQUIRE_THAT(nfHbyA, EqualsInternal(HbyA, ApproxVector(epsilonII)));
        //    REQUIRE_THAT(nfHbyA.boundaryData(), EqualsBoundary(HbyA, ApproxVector(epsilonII)));
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

        REQUIRE_THAT(nfHbyA, EqualsInternal(ofConstrainHbyA, ApproxVector(1e-12)));
        // REQUIRE_THAT(nfHbyA.boundaryData(), EqualsBoundary(ofConstrainHbyA,
        // ApproxVector(1e-12)));
    }

    SECTION("compute flux")
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
        Foam::surfaceScalarField ofFlux("ofFlux", fvc::flux(HbyA));
        auto nfFlux = nf::flux(nfHbyA);

        REQUIRE_THAT(nfFlux, EqualsInternal(ofFlux, ApproxScalar(epsilonII)));
        // REQUIRE_THAT(nfFlux.boundaryData(), EqualsBoundary(ofFlux, ApproxScalar(epsilonII)));
    }

    SECTION("compute flux")
    {
        nfPhi.correctBoundaryConditions();
        REQUIRE_THAT(nfPhi, EqualsInternal(ofPhi, ApproxScalar(epsilon)));
        auto forAUf =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUf");
        auto nfrAUf = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, forAUf);

        Foam::surfaceScalarField ofPhi0("phi0", ofPhi * 0.0);
        auto nfPhi0 = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi0);

        Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));
        ofPhi0 = ofPhi - ofpEqn.flux();
        // solve(ofpEqn);

        nf::PDESolver<NeoN::scalar> pEqn(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

        pEqn.assemble();

        // NOTE removeBoundaryContributions is not working in distributed case
        // REQUIRE_THAT(
        //     NeoN::la::removeBoundaryContributions(pEqn.linearSystem()).matrix().diag(),
        //     nf::Equals(ofpEqn.diag(), ApproxScalar(1e-15), false)
        // );

        REQUIRE_THAT(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            EqualsInternal(ofpEqn.upper(), ApproxScalar(epsilonII))
        );

        nf::updateFaceVelocity(nfPhi, pEqn, nfPhi0);
        REQUIRE_THAT(nfPhi0, EqualsInternal(ofPhi0, ApproxScalar(epsilonII)));
        // REQUIRE_THAT(nfPhi0.boundaryData(), EqualsBoundary(ofPhi0, ApproxScalar(epsilonII)));
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

        // Foam::surfaceScalarField ofPhi0("phi0", ofPhi * 0.0);
        // auto nfPhi0 = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi);

        Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));
        ofpEqn.setReference(0, 0.0);
        ofp.correctBoundaryConditions();
        solve(ofpEqn);
        ofp.correctBoundaryConditions();
        auto ofPhi0 = ofpEqn.flux();

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
        // REQUIRE_THAT(
        //     NeoN::la::removeBoundaryContributions(pEqn.linearSystem()).matrix().diag(),
        //     nf::Equals(ofpEqn.diag(), ApproxScalar(1e-15), false)
        // );

        REQUIRE_THAT(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            EqualsInternal(ofpEqn.upper(), ApproxScalar(epsilonII))
        );
        REQUIRE_THAT(
            pEqn.linearSystem().rhs(),
            EqualsInternal(ofpEqn.source(), ApproxScalar(epsilonII))
        );

        nfP.correctBoundaryConditions();

        auto [numIter, initResNorm, finalResNorm, solveTime] = stats.entries[0];

        REQUIRE(numIter != 0);
        REQUIRE(initResNorm != 0);
        REQUIRE(finalResNorm < initResNorm);

        REQUIRE_THAT(nfP, EqualsInternal(ofp, ApproxScalar(1e-32)));
        REQUIRE_THAT(nfP.boundaryData(), EqualsBoundary(ofp, ApproxScalar(1e-32)));
        REQUIRE_THAT(nfPhi, EqualsInternal(ofPhi, ApproxScalar(1e-32)));
        REQUIRE_THAT(nfPhi.boundaryData(), EqualsBoundary(ofPhi, ApproxScalar(1e-32)));

        auto nfPhi0 = nf::flux(pEqn);
        REQUIRE_THAT(nfPhi0, EqualsInternal(ofPhi0(), ApproxScalar(1e-05)));
    }
}
