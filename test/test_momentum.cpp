// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"
#include "constrainHbyA.H"

using Foam::Info;
using Foam::endl;
using Foam::nl;
namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr; // A single time object


TEST_CASE("Momentum")
{
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

    SECTION("Solve transient momentum without grad(p)")
    {
        // require fields to be initially the same
        nf::compare(nfU, ofU, ApproxVector({1e-15, 1e-15, 1e-15}));

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

        Foam::solve(ofUEqn);

        nfUEqn.solve();
        nf::compare(nfU, ofU, ApproxVector({1e-06, 1e-02, 1e-02}));
    }

    // SECTION("Solve transient momentum with grad(p)")
    // {
    //     auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    //     solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

    //     // require fields to be initially the same
    //     auto hostnfU = nfU.internalVector().copyToHost();
    //     auto hostnfP = nfP.internalVector().copyToHost();
    //     for (size_t celli = 0; celli < hostnfU.size(); celli++)
    //     {
    //         REQUIRE(hostnfU.view()[celli][0] == Catch::Approx(ofU[celli][0]).margin(1e-12));
    //         REQUIRE(hostnfU.view()[celli][1] == Catch::Approx(ofU[celli][1]).margin(1e-12));
    //         // REQUIRE(hostnfU.view()[celli][2] ==
    //         Catch::Approx(ofU[celli][2]).margin(1e-12)); REQUIRE(hostnfP.view()[celli] ==
    //         Catch::Approx(ofp[celli]).margin(1e-12));
    //     }

    //     Foam::fvVectorMatrix ofUEqn(
    //         Foam::fvm::ddt(ofU) + Foam::fvm::div(ofPhi, ofU) - Foam::fvm::laplacian(ofNu,
    //         ofU)
    //     );

    //     Foam::solve(ofUEqn == -Foam::fvc::grad(ofp));

    //     nf::PDESolver<NeoN::Vec3> nfUEqn(
    //         dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
    //         nfU,
    //         rt
    //     );

    //     nfUEqn.solve(-1.0 * dsl::exp::grad(nfP));

    //     auto hostnfU2 = nfU.internalVector().copyToHost();
    //     for (size_t celli = 0; celli < hostnfU.size(); celli++)
    //     {
    //         REQUIRE(hostnfU2.view()[celli][0] == Catch::Approx(ofU[celli][0]).margin(1e-12));
    //         // NOTE this test seems to be prone to
    //         // https://github.com/catchorg/Catch2/issues/1863
    //         REQUIRE(hostnfU2.view()[celli][1]
    //         // == Catch::Approx(ofU[celli][1]).margin(1e-12)); NOTE we lower the criterion
    //         here
    //         // because OF explicitly zeros in the 2D case
    //         REQUIRE(hostnfU2.view()[celli][2] == Catch::Approx(ofU[celli][2]).margin(1e-06));
    //     }

    //     // NOTE we here test that that rAU and HbyA are correct
    //     // regardless how UEqn was formulated, with or without grad(p)
    //     // A better way is to just test correctness of computeRAUandHbyA once
    //     // and check whether solve(grad(p)) does not modify the original matrix
    //     // in UEqn
    //     SECTION("HbyA modified U")
    //     {
    //         ofU.correctBoundaryConditions();
    //         Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
    //         Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());

    //         nfU.correctBoundaryConditions();
    //         auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);

    //         auto hostnfrAU = nfrAU.internalVector().copyToHost();
    //         for (size_t celli = 0; celli < hostnfrAU.size(); celli++)
    //         {
    //             REQUIRE(hostnfrAU.view()[celli] ==
    //             Catch::Approx(forAU[celli]).margin(1e-12));
    //         }

    //         auto hostnfHbyA = nfHbyA.internalVector().copyToHost();
    //         for (size_t celli = 0; celli < hostnfHbyA.size(); celli++)
    //         {
    //             REQUIRE(
    //                 hostnfHbyA.view()[celli][0] == Catch::Approx(HbyA[celli][0]).margin(1e-8)
    //             );
    //             REQUIRE(
    //                 hostnfHbyA.view()[celli][1] == Catch::Approx(HbyA[celli][1]).margin(1e-8)
    //             );
    //             // REQUIRE(
    //             //     hostnfHbyA.view()[celli][2] ==
    //             Catch::Approx(HbyA[celli][2]).margin(1e-8)
    //             // );
    //         }

    //         Foam::surfaceScalarField phiHbyA("phiHbyA", Foam::fvc::flux(HbyA));

    //         auto nfPhiHbyA = nf::flux(nfHbyA);
    //         auto hostnfPhiHbyA = nfPhiHbyA.internalVector().copyToHost();
    //         for (size_t celli = 0; celli < hostnfHbyA.size(); celli++)
    //         {
    //             REQUIRE(
    //                 hostnfPhiHbyA.view()[celli] == Catch::Approx(phiHbyA[celli]).margin(1e-6)
    //             );
    //         }
    //     }
    // }
}
