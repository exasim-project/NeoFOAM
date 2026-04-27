
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
    // ofU.correctBoundaryConditions();
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
        nf::compare(nfU, ofU, ApproxVector(epsilon), true);

        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        forAU.correctBoundaryConditions();
        nfUEqn.assemble();
        auto nfrAU = nf::computeRAU(nfUEqn);

        NeoFOAM::compare(nfrAU, forAU, ApproxScalar(1e-15), true);
    }

    SECTION("HbyA" + execName)
    {
        nf::compare(nfU, ofU, ApproxVector(epsilon), true);
        nf::compare(nfPhi, ofPhi, ApproxScalar(epsilon), true);
        nf::compare(nfUEqn.linearSystem().rhs(), ofUEqn.source(), ApproxVector(epsilon), true);

        Foam::volScalarField forAU("rAU", 1.0 / ofUEqn.A());
        Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());

        nfUEqn.assemble();

        nf::compare(
            NeoN::la::removeBoundaryContributions(nfUEqn.linearSystem()).matrix().diag(),
            ofUEqn.diag(),
            ApproxVector(1e-15)
        );

        nf::compare(
            NeoN::la::upper(nfUEqn.linearSystem().matrix()),
            ofUEqn.upper(),
            ApproxVector(1e-15)
        );

        auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);

        // FIXME this needs very lose tolerance to pass
        // SECTION_IF(rt.mpiEnvironment.rank() == 0, "Correct boundaryMesh on rank 0")
        // {
        //     nf::compare(nfHbyA, HbyA, ApproxVector({1e-01, 1e-01, 1e-01}), false);
        // }
        // SECTION_IF(rt.mpiEnvironment.rank() == 2, "Correct boundaryMesh on rank 2")
        // {
        // nf::compare(nfHbyA, HbyA, ApproxVector({1e-01, 1e-01, 1e-01}), false);
        // }
        // FIXME This fails
        // SECTION_IF(rt.mpiEnvironment.rank() == 1, "Correct boundaryMesh on !rank 1")
        // {
        // nf::compare(nfHbyA, HbyA, ApproxVector({1e-01, 1e-01, 1e-01}), false);
        // }
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

        // SECTION_IF(rt.mpiEnvironment.rank() == 0, "Correct boundaryMesh on !rank 1")
        // {
        //     nf::compare(nfHbyA, ofConstrainHbyA, ApproxVector({1e-01, 1e-01, 1e-01}), false);
        // }
    }

    SECTION("compute flux")
    {
        nf::compare(nfPhi, ofPhi, ApproxScalar(epsilon), false);
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

        pEqn.assemble();

        nf::compare(
            NeoN::la::removeBoundaryContributions(pEqn.linearSystem()).matrix().diag(),
            ofpEqn.diag(),
            ApproxScalar(1e-15),
            false
        );

        nf::compare(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            ofpEqn.upper(),
            ApproxScalar(1e-15),
            false
        );

        nf::updateFaceVelocity(nfPhi, pEqn, nfPhi0);
        nf::compare(nfPhi0, ofPhi0, ApproxScalar(1e-32), true);
    }
}
