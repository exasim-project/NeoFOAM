// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// OpenFOAM's "bounded <scheme>" convection prefix (Foam::fv::boundedConvectionScheme) subtracts
// fvm::Sp(fvc::surfaceIntegrate(faceFlux), psi) from the inner scheme's discretisation. NeoN
// implements the same wrapper (BoundedDiv), so the compatibility layer must hand the prefix
// through untouched — dropping it makes the NeoN side solve a different steady-state equation.
//
// The scheme lives on disk as `div(boundedPhi,U) bounded Gauss upwind;` in the shared
// setup_pressureVelocityCoupling fvSchemes; naming the flux field `boundedPhi` lets both backends
// select it from that one entry, so OpenFOAM's own bounded scheme is the oracle. The flux is
// fvc::flux(U) of a random U and is therefore not divergence free — without the Sp term the two
// matrices differ by ~5% of the diagonal, far above the 1e-8 comparison margin.

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr; // A single time object


TEST_CASE("mapFvSchemes keeps the bounded div scheme prefix")
{
    Foam::Time& runTime = *timePtr;

    auto rt = nf::createAdapterRunTime(runTime, NeoN::SerialExecutor {});
    auto mapped = nf::mapFvSchemes(rt.fvSchemesDict);

    auto& tokens = mapped.subDict("divSchemes").get<NeoN::TokenList>("div(boundedPhi,U)");

    REQUIRE(tokens.size() == 3u);
    REQUIRE(tokens.get<std::string>(0) == "bounded");
    REQUIRE(tokens.get<std::string>(1) == "Gauss");
    REQUIRE(tokens.get<std::string>(2) == "upwind");
}

TEST_CASE("BoundedDivScheme")
{
    float epsilon = 1e-32;
    Foam::Time& runTime = *timePtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);

    auto ofU = randomVectorField(runTime, mesh, "U");
    ofU.correctBoundaryConditions();
    auto& oldOfU = ofU.oldTime();
    oldOfU.primitiveFieldRef() = Foam::vector(0.0, 0.0, 0.0);
    oldOfU.correctBoundaryConditions();

    Foam::surfaceScalarField ofPhi(
        Foam::IOobject(
            "boundedPhi",
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

    auto& vectorCollection = nnfvcc::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfU = NeoFOAM::constructAndRegister(vectorCollection, rt, ofU);
    auto& nfOldU = fvcc::oldTime(nfU);
    NeoN::fill(nfOldU.internalVector(), NeoN::Vec3(0.0, 0.0, 0.0));
    nfOldU.correctBoundaryConditions();

    auto nfPhi = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi);
    auto nfNu = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofNu);

    SECTION("Solve momentum with a bounded div scheme on " + execName)
    {
        Foam::fvVectorMatrix ofUEqn(
            fvm::ddt(ofU) + fvm::div(ofPhi, ofU) - fvm::laplacian(ofNu, ofU)
        );

        nf::PDE<NeoN::Vec3> nfUEqn(
            dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
            nfU,
            rt
        );

        NeoN::fill(nfUEqn.linearSystem().rhs(), NeoN::Vec3(0.0, 0.0, 0.0));

        // require fields to be initially the same
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector(epsilon)));

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

        Foam::solve(ofUEqn);
        nfUEqn.setFinalIter(true);
        nfUEqn.solve();

        nfU.correctBoundaryConditions();
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector({1e-08, 1e-08, 1e-08})));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector({1e-08, 1e-08, 1e-08})));
    }
}
