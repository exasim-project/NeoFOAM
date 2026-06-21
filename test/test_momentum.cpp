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


TEST_CASE("Momentum")
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

    auto& nfU = NeoFOAM::constructAndRegister(vectorCollection, rt, ofU);
    auto& nfOldU = fvcc::oldTime(nfU);
    NeoN::fill(nfOldU.internalVector(), NeoN::Vec3(0.0, 0.0, 0.0));
    nfOldU.correctBoundaryConditions();

    SECTION("Solve transient momentum without grad(p) on " + execName)
    {
        Foam::fvVectorMatrix ofUEqn(
            fvm::ddt(ofU) + fvm::div(ofPhi, ofU) - fvm::laplacian(ofNu, ofU)
        );

        nf::PDESolver<NeoN::Vec3> nfUEqn(
            dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
            nfU,
            rt
        );

        NeoN::fill(nfUEqn.linearSystem().rhs(), NeoN::Vec3(0.0, 0.0, 0.0));

        // require fields to be initially the same
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfP, EqualsInternal(ofp, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfP.boundaryData(), EqualsBoundary(ofp, ApproxScalar(epsilon)));

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

        Foam::solve(ofUEqn);
        nfUEqn.setFinalIter(true);
        nfUEqn.solve();

        nfU.correctBoundaryConditions();
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector({1e-08, 1e-08, 1e-08})));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector({1e-08, 1e-08, 1e-08})));
    }

    SECTION("Solve transient momentum with grad(p) on " + execName)
    {
        Foam::fvVectorMatrix ofUEqn(
            fvm::ddt(ofU) + fvm::div(ofPhi, ofU) - fvm::laplacian(ofNu, ofU)
        );

        nf::PDESolver<NeoN::Vec3> nfUEqn(
            dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
            nfU,
            rt
        );

        NeoN::fill(nfUEqn.linearSystem().rhs(), NeoN::Vec3(0.0, 0.0, 0.0));

        // require fields to be initially the same
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfP, EqualsInternal(ofp, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfP.boundaryData(), EqualsBoundary(ofp, ApproxScalar(epsilon)));

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

        Foam::solve(ofUEqn == -fvc::grad(ofp));

        nfUEqn.setFinalIter(true);
        nfUEqn.solve(-1.0 * dsl::exp::grad(nfP));
        nfU.correctBoundaryConditions();
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector({1e-08, 1e-08, 1e-08})));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector({1e-08, 1e-08, 1e-08})));
    }
}
