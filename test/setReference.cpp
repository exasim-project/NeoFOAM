// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Dedicated test for the setReference / setRefCell logic in PDESolver.
// Uses setup_ddtCorr which has all-Neumann pressure BCs (movingWall and
// fixedWalls are zeroGradient; frontAndBack is empty) so `p.needReference()`
// is true and the setRefCell -> setReference path must be exercised for the
// pressure equation to have a unique solution.

#define CATCH_CONFIG_RUNNER

#include "common.hpp"
#include "findRefCell.H"

namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace nf = NeoFOAM;

using NeoFOAM::EqualsBoundary;
using NeoFOAM::EqualsInternal;

extern Foam::Time* timePtr;

TEST_CASE("PressureSetReference")
{
    float epsilon = 1e-15;
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);

    // Read the case fields. p is all-Neumann here -> needReference() = true.
    auto ofU = NeoFOAM::randomVectorField(runTime, mesh, "U");
    auto ofp = NeoFOAM::randomScalarField(runTime, mesh, "p");
    ofp.correctBoundaryConditions();
    ofU.correctBoundaryConditions();

    auto& vectorCollection =
        NeoN::finiteVolume::cellCentred::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfP = NeoFOAM::constructAndRegister(vectorCollection, rt, ofp, false);

    // Map the p solver entry from OF syntax to NeoN format.
    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

    SECTION("setReference for all-Neumann pressure on " + execName)
    {
        REQUIRE(ofp.needReference());

        // Build a physically meaningful, divergence-free phi from a uniform
        // velocity U=(1,0,0). The discrete divergence of this flux sums to
        // zero per cell on the Cartesian setup_ddtCorr mesh, so div(phi) is
        // compatible with the all-Neumann pressure system and the pinned
        // solution is unique modulo the soft-pin null mode.
        ofU.primitiveFieldRef() = Foam::vector(1.0, 0.0, 0.0);
        ofU.correctBoundaryConditions();
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
        auto nfPhi = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi);

        // A random positive rAUf (laplacian coefficient). The field name
        // must match an entry in setup_ddtCorr's fvSchemes laplacianSchemes
        // — that entry is "laplacian(rAUfNF,p)".
        auto forAUf =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUfNF");
        auto nfrAUf = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, forAUf);

        // Resolve the global reference cell exactly the way neoIcoFoam does;
        // setRefCell makes pRefCell >= 0 only on the rank that owns the
        // global reference cell (rank 0 in this serial setup).
        Foam::label pRefCell = 0;
        Foam::scalar pRefValue = 0.0;
        Foam::setRefCell(ofp, ofp.mesh().solutionDict().subDict("PISO"), pRefCell, pRefValue);

        Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));
        if (ofp.needReference() && pRefCell >= 0)
        {
            ofpEqn.setReference(pRefCell, pRefValue);
        }
        solve(ofpEqn);

        nf::PDESolver<NeoN::scalar> pEqn(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );
        if (ofp.needReference() && pRefCell >= 0)
        {
            pEqn.setReference(static_cast<NeoN::localIdx>(pRefCell), pRefValue);
        }

        auto stats = pEqn.solve();

        // Matrix assembled with the diag doubled at pRefCell (`setReference`
        // is mathematically diag *= 2, rhs += diag*pRefValue on the pinned
        // row) — should match OF's fvMatrix::setReference exactly.
        REQUIRE_THAT(
            pEqn.linearSystem().matrix().diag(),
            EqualsInternal(ofpEqn.diag(), ApproxScalar(epsilon))
        );
        REQUIRE_THAT(
            NeoN::la::upper(pEqn.linearSystem().matrix()),
            EqualsInternal(ofpEqn.upper(), ApproxScalar(epsilon))
        );
        REQUIRE_THAT(
            pEqn.linearSystem().rhs(),
            EqualsInternal(ofpEqn.source(), ApproxScalar(epsilon))
        );

        auto [numIter, initResNorm, finalResNorm, solveTime] = stats.entries[0];
        REQUIRE(numIter != 0);
        REQUIRE(initResNorm != 0);
        REQUIRE(finalResNorm < initResNorm);

        REQUIRE_THAT(nfP, EqualsInternal(ofp, ApproxScalar(1e-12)));
    }
}
