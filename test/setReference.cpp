// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Dedicated test for the setReference / setRefCell logic in PDE.
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

        nf::PDE<NeoN::scalar> pEqn(
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

        // Regression test for the ELL PDE + Solver + SetReference::applyELL() path:
        // same all-Neumann pressure system, but assembled/solved through
        // PDE<scalar, scalar, localIdx, ELLMatrix<...>> and its matching Solver via
        // solveWith (the path neoIcoFoam.cpp will eventually use for pEqn), not the
        // solve()/solveImpl path the CSR check above exercises.
        SECTION("ELL matches CSR and OpenFOAM on " + execName)
        {
            using EllMatrix = NeoN::la::ELLMatrix<NeoN::scalar, NeoN::localIdx>;

            // Own solverDict/linearSystem-cache entries, distinct from "p" above --
            // PDE/Solver key both off VolumeField::name, so a shared name would collide
            // with the CSR system already cached under RunTime's "linearSystemp".
            solverDict.insert("pEll", solverDict.subDict("p"));
            auto nfPEll = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofp);
            nfPEll.name = "pEll";
            nfPEll.correctBoundaryConditions();

            nf::PDE<NeoN::scalar, NeoN::scalar, NeoN::localIdx, EllMatrix> pEqnEll(
                dsl::imp::laplacian(nfrAUf, nfPEll) - dsl::exp::div(nfPhi),
                nfPEll,
                rt
            );
            if (ofp.needReference() && pRefCell >= 0)
            {
                pEqnEll.setReference(static_cast<NeoN::localIdx>(pRefCell), pRefValue);
            }

            nf::Solver<NeoN::scalar, NeoN::scalar, NeoN::localIdx, EllMatrix> pSolverEll(
                nfPEll,
                rt
            );
            auto statsEll = pSolverEll.solve(pEqnEll);

            REQUIRE_THAT(
                pEqnEll.linearSystem().matrix().diag(),
                EqualsInternal(ofpEqn.diag(), ApproxScalar(epsilon))
            );
            REQUIRE_THAT(
                pEqnEll.linearSystem().rhs(),
                EqualsInternal(ofpEqn.source(), ApproxScalar(epsilon))
            );

            // NeoN::la::upper() has no ELL overload, so the off-diagonal (neighbour)
            // coefficients can't be checked the same way as the CSR case above. Probe them
            // instead via computeResidual (format-generic): apply pEqnEll's own matrix/rhs to
            // its own converged solution -- if the off-diagonals (and the solve itself) are
            // correct, A_ell * nfPEll - b_ell should be tiny.
            NeoN::Vector<NeoN::scalar> residualEll(rt.exec, nfPEll.mesh().nCells(), 0.0);
            NeoN::la::computeResidual(
                pEqnEll.linearSystem().matrix(),
                pEqnEll.linearSystem().rhs(),
                nfPEll.internalVector(),
                residualEll
            );
            auto residualHost = residualEll.copyToHost();
            auto residualView = residualHost.view();
            NeoN::scalar maxAbsResidual = 0.0;
            for (NeoN::localIdx i = 0; i < residualView.size(); ++i)
            {
                maxAbsResidual = std::max(maxAbsResidual, std::abs(residualView[i]));
            }
            REQUIRE(maxAbsResidual < 1e-8);

            auto [numIterEll, initResNormEll, finalResNormEll, solveTimeEll] = statsEll.entries[0];
            REQUIRE(numIterEll != 0);
            REQUIRE(initResNormEll != 0);
            REQUIRE(finalResNormEll < initResNormEll);

            // Loose tolerance (vs. 1e-12 for the CSR check above): this is a pure-Neumann
            // system regularized only by a single-cell reference pin (diag *= 2), which is
            // weakly conditioned near its suppressed null space. CSR and ELL walk the same
            // matrix entries (verified above) but in different order, so Cg+Jacobi's
            // floating-point summation differs and the two converge to slightly different
            // points on the solution manifold -- each independently valid (see the
            // self-consistency residual check above), just not bit-identical.
            REQUIRE_THAT(nfPEll, EqualsInternal(ofp, ApproxScalar(1e-4)));
        }
    }
}
