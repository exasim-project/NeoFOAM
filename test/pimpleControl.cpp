// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Unit tests for the NeoFOAM-native PimpleControl outer-loop control.
// The control is unit-testable WITHOUT a solve: construct PimpleControl from a hand-built
// NeoN::Dictionary (mirroring what convert(fvSolution) produces) and drive loop() with
// synthetic {initResNorm, finalResNorm} residual maps. One section additionally loads the
// setup_pimple case to prove the real fvSolution.PIMPLE block reads cleanly.

#define CATCH_CONFIG_RUNNER

#include "common.hpp"
#include "findRefCell.H"
#include "NeoFOAM/solutionControl/pimpleControl.hpp"

namespace fvc = Foam::fvc;
namespace dsl = NeoN::dsl;
namespace nf = NeoFOAM;

// Provided by the shared catch2 main (the single Foam::Time object); used by the
// [finaliterselect] integration section which loads the setup_pimple case and solves.
extern Foam::Time* timePtr;

namespace
{

// Build a PIMPLE-only fvSolution dict inline. nOuterCorrectors is inserted as a NeoN::label
// (what convert(Foam::label) yields), matching the reader's get<NeoN::label>.
NeoN::Dictionary makePimpleDict(NeoN::label nOuterCorrectors)
{
    NeoN::Dictionary pimple;
    pimple.insert("nOuterCorrectors", nOuterCorrectors);
    NeoN::Dictionary fvSolution;
    fvSolution.insert("PIMPLE", pimple);
    return fvSolution;
}

// Add a residualControl { <field> { tolerance; relTol; } } sub-dict to a PIMPLE block.
// Tolerances are inserted via the supplied any so callers can exercise both scalar and
// bare-int (label) typed entries (the int-tolerant asScalar path).
void addResidualControl(
    NeoN::Dictionary& fvSolution,
    const std::string& field,
    const std::any& tolerance,
    const std::any& relTol
)
{
    auto& pimple = fvSolution.subDict("PIMPLE");
    if (!pimple.contains("residualControl"))
    {
        pimple.insert("residualControl", NeoN::Dictionary {});
    }
    auto& rc = pimple.subDict("residualControl");
    NeoN::Dictionary fieldDict;
    fieldDict.insert("tolerance", tolerance);
    fieldDict.insert("relTol", relTol);
    rc.insert(field, fieldDict);
}

} // namespace

TEST_CASE("PimpleControl loop count", "[pimpleControl][loopcount]")
{
    SECTION("nOuterCorrectors == 3 with no residualControl iterates exactly 3 times")
    {
        auto dict = makePimpleDict(3);
        nf::PimpleControl pimple(dict);
        REQUIRE(pimple.nOuterCorrectors() == 3);

        int passes = 0;
        while (pimple.loop({}))
        {
            ++passes;
            REQUIRE(passes <= 3); // guard against a runaway loop
        }
        REQUIRE(passes == 3);

        // corr_ has reset after exit: the control is reusable for a second time step.
        int passes2 = 0;
        while (pimple.loop({}))
        {
            ++passes2;
            REQUIRE(passes2 <= 3);
        }
        REQUIRE(passes2 == 3);
    }

    SECTION("no PIMPLE block defaults to nOuterCorrectors == 1 (one pass)")
    {
        NeoN::Dictionary empty;
        nf::PimpleControl pimple(empty);
        REQUIRE(pimple.nOuterCorrectors() == 1);

        int passes = 0;
        while (pimple.loop({}))
        {
            ++passes;
            REQUIRE(passes <= 1);
        }
        REQUIRE(passes == 1);
    }

    SECTION("setup_pimple-shaped PIMPLE block reads cleanly (nOuterCorrectors 3, bare-int relTol)")
    {
        // Mirrors test/setup_pimple/system/fvSolution after convert(): nOuterCorrectors as a
        // label and a residualControl{ p{ tolerance 1e-6; relTol 0; } U{...} } sub-dict where
        // `relTol 0;` is a bare integer (int-vs-scalar typing risk). The control reads it
        // via the int-tolerant asScalar path without bad_any_cast. This is intentionally an
        // inline dict, not a loaded RunTime: building a second MeshAdapter here would leak
        // Kokkos-backed allocations torn down after NeoN::finalize() (a teardown SIGSEGV).
        auto dict = makePimpleDict(3);
        addResidualControl(dict, "p", NeoN::scalar(1e-6), int(0));
        addResidualControl(dict, "U", NeoN::scalar(1e-6), int(0));
        nf::PimpleControl pimple(dict);
        REQUIRE(pimple.nOuterCorrectors() == 3);

        int passes = 0;
        while (pimple.loop({})) // no residuals fed -> runs the full count
        {
            ++passes;
            REQUIRE(passes <= 3);
        }
        REQUIRE(passes == 3);
    }
}

TEST_CASE("PimpleControl residual exit", "[pimpleControl][residualexit]")
{
    SECTION("absolute criterion drives a two-phase converged exit in < 50 passes")
    {
        auto dict = makePimpleDict(50);
        addResidualControl(dict, "p", NeoN::scalar(1e-6), NeoN::scalar(0.0)); // tolerance, relTol
        nf::PimpleControl pimple(dict);

        // From pass 2 onward the pressure final residual is well below the abs tolerance.
        nf::ResidualMap residuals;
        residuals["p"] = {1.0, 1e-8}; // {initResNorm, finalResNorm}

        int passes = 0;
        bool finalIterOnLastPass = false;
        while (pimple.loop(residuals))
        {
            ++passes;
            finalIterOnLastPass = pimple.finalIter();
            REQUIRE(passes < 50); // must exit well before exhausting the count
        }
        // pass 1: skip (corr_==1). pass 2: store initial. pass 3: criteria met -> converged_
        // set, ONE more pass runs (pass 4, finalIter() true). pass 5 -> loop() returns false.
        REQUIRE(passes < 50);
        REQUIRE(finalIterOnLastPass); // the two-phase converged final pass fired
    }

    SECTION("relative criterion converges (final/initial below relTol)")
    {
        auto dict = makePimpleDict(50);
        addResidualControl(dict, "p", NeoN::scalar(0.0), NeoN::scalar(0.01)); // abs off, relTol
        nf::PimpleControl pimple(dict);

        // Stored initial p residual 1.0 on pass 2; final 0.005 -> 0.005/1.0 = 0.005 < 0.01.
        nf::ResidualMap residuals;
        residuals["p"] = {1.0, 0.005};

        int passes = 0;
        while (pimple.loop(residuals))
        {
            ++passes;
            REQUIRE(passes < 50);
        }
        REQUIRE(passes < 50);
    }

    SECTION("convergence is never checked on the first outer iteration")
    {
        auto dict = makePimpleDict(50);
        addResidualControl(dict, "p", NeoN::scalar(1e-6), NeoN::scalar(0.0));
        nf::PimpleControl pimple(dict);

        // Pre-load criteria-satisfying residuals: the first loop() call must still return true
        // (corr_ == 1 skip) regardless of the fed residuals.
        nf::ResidualMap residuals;
        residuals["p"] = {1.0, 1e-12};
        REQUIRE(pimple.loop(residuals) == true);
        REQUIRE(pimple.firstIter()); // corr_ == 1 on the first pass
    }
}

TEST_CASE("PimpleControl finalIter", "[pimpleControl][finaliter]")
{
    SECTION("finalIter() false on passes 1..n-1, true on pass n (count-driven)")
    {
        auto dict = makePimpleDict(3);
        nf::PimpleControl pimple(dict);

        REQUIRE(pimple.loop({})); // pass 1
        REQUIRE_FALSE(pimple.finalIter());
        REQUIRE(pimple.loop({})); // pass 2
        REQUIRE_FALSE(pimple.finalIter());
        REQUIRE(pimple.loop({})); // pass 3 (== nOuterCorrectors)
        REQUIRE(pimple.finalIter());
        REQUIRE_FALSE(pimple.loop({})); // pass 4 -> stop
    }

    SECTION("bare-int `tolerance 0;` residualControl constructs without throwing")
    {
        auto dict = makePimpleDict(3);
        // int(0) mirrors a bare-integer `tolerance 0;` token (stored as NeoN::label==int).
        addResidualControl(dict, "p", int(0), int(0));
        REQUIRE_NOTHROW(nf::PimpleControl(dict));

        // And it still drives the count when residuals never satisfy a zero abs/rel tol.
        nf::PimpleControl pimple(dict);
        nf::ResidualMap residuals;
        residuals["p"] = {1.0, 1.0};
        int passes = 0;
        while (pimple.loop(residuals))
        {
            ++passes;
            REQUIRE(passes <= 3);
        }
        REQUIRE(passes == 3);
    }
}

// Integration test proving the solveImpl <field>Final-subdict selection deterministically:
// when setFinalIter(true) is set AND a pFinal solver subdict exists, the final-pass solve uses the
// pFinal configuration; when setFinalIter(false), it uses base p. The two subdicts are made
// OBSERVABLY different via the relative-residual tolerance: base p relTol 0.5 (loose -> few
// iterations) vs pFinal relTol 1e-10 (tight -> many more iterations on the IDENTICAL system). The
// committed observable is the integer iteration count (SolverStatsEntry::numIter) — a strict
// inequality with no floating-point margin, so it does not depend on solver-convergence
// nondeterminism. setup_pimple is the closed-domain all-Neumann fixture (p.needReference() == true,
// ref cell read from the PIMPLE subdict, mirroring neoPimpleFoam).
TEST_CASE(
    "PDESolver finalIter selects field-Final solver subdict",
    "[pimpleControl][finaliterselect]"
)
{
    Foam::Time& runTime = *timePtr;
    // Serial executor only: the count-difference observable is deterministic and the assertion is
    // executor-independent; one executor keeps the integration solve cheap and avoids a second
    // MeshAdapter teardown path.
    auto exec = NeoN::Executor(NeoN::SerialExecutor {});

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    rt.fvSchemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

    // All-Neumann pressure fixture -> needReference() is true (closed domain).
    auto ofU = nf::randomVectorField(runTime, mesh, "U");
    auto ofp = nf::randomScalarField(runTime, mesh, "p");
    ofp.correctBoundaryConditions();
    ofU.correctBoundaryConditions();

    auto& vectorCollection =
        NeoN::finiteVolume::cellCentred::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfP = nf::constructAndRegister(vectorCollection, rt, ofp, false);

    // Uniform U=(1,0,0) -> divergence-free phi on the Cartesian fixture, compatible with the
    // all-Neumann pressure system (mirrors test_setReference's construction).
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
    auto nfPhi = nf::constructFrom(rt.exec, rt.nfMesh, ofPhi);

    // Random positive laplacian coefficient; the field name must match setup_pimple's
    // laplacianSchemes entry "laplacian(rAUfNF,p)".
    auto forAUf = nf::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUfNF");
    auto nfrAUf = nf::constructFrom(rt.exec, rt.nfMesh, forAUf);

    // Reference cell from the PIMPLE subdict (neoPimpleFoam reads PIMPLE, not PISO).
    Foam::label pRefCell = 0;
    Foam::scalar pRefValue = 0.0;
    Foam::setRefCell(ofp, ofp.mesh().solutionDict().subDict("PIMPLE"), pRefCell, pRefValue);

    // Build a fresh pEqn over the same expression/field and solve with a given finalIter flag.
    // A fresh solver per pass guarantees the only difference between passes is finalIter_.
    auto solvePass = [&](bool finalIter) -> int
    {
        nf::PDESolver<NeoN::scalar> pEqn(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );
        if (ofp.needReference() && pRefCell >= 0)
        {
            pEqn.setReference(static_cast<NeoN::localIdx>(pRefCell), pRefValue);
        }
        pEqn.setFinalIter(finalIter);
        auto stats = pEqn.solve();
        return stats.entries[0].numIter;
    };

    SECTION("setFinalIter(true) selects the tighter pFinal subdict (more iterations than base p)")
    {
        REQUIRE(ofp.needReference());

        // Make base p and pFinal OBSERVABLY different BEFORE mapping: base p relTol 0.5 (loose),
        // pFinal relTol 1e-10 (tight). Identical large maxIter (5000) keeps the tight case from
        // being iteration-capped before it converges further. tolerance stays 0 (abs off) so the
        // relative-residual stop is the sole differentiator.
        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        REQUIRE(solverDict.isDict("pFinal"));

        auto& pDict = solverDict.subDict("p");
        pDict.insert("relTol", NeoN::scalar(0.5));
        pDict.insert("tolerance", NeoN::scalar(0.0));
        pDict.insert("maxIter", NeoN::label(5000));

        auto& pFinalDict = solverDict.subDict("pFinal");
        pFinalDict.insert("relTol", NeoN::scalar(1e-10));
        pFinalDict.insert("tolerance", NeoN::scalar(0.0));
        pFinalDict.insert("maxIter", NeoN::label(5000));

        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));
        solverDict.subDict("pFinal") = nf::mapFvSolution(solverDict.subDict("pFinal"));

        const int numIterBase = solvePass(false); // selects base p (relTol 0.5)
        const int numIterFinal = solvePass(true); // selects pFinal (relTol 1e-10)

        REQUIRE(numIterBase > 0);
        // The tighter pFinal relTol forces strictly MORE solver iterations than the loose base
        // p relTol on the identical system -> the selected subdict is detectable from numIter.
        REQUIRE(numIterFinal > numIterBase);
    }

    SECTION("absent pFinal -> setFinalIter(true) is a no-op (identical iteration count)")
    {
        REQUIRE(ofp.needReference());

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        // Remove pFinal so the isDict guard must fall back to base p on the final pass.
        if (solverDict.contains("pFinal"))
        {
            solverDict.remove("pFinal");
        }
        REQUIRE_FALSE(solverDict.isDict("pFinal"));

        auto& pDict = solverDict.subDict("p");
        pDict.insert("relTol", NeoN::scalar(0.5));
        pDict.insert("tolerance", NeoN::scalar(0.0));
        pDict.insert("maxIter", NeoN::label(5000));
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

        const int numIterBaseNoPFinal = solvePass(false); // base p
        const int numIterFinalNoPFinal = solvePass(true); // no pFinal -> falls back to base p

        // No pFinal subdict -> finalIter selection is a no-op: identical solver config -> identical
        // iteration count on the identical system.
        REQUIRE(numIterFinalNoPFinal == numIterBaseNoPFinal);
    }
}

// The setReference rank-0 guard, re-applied on every pressure solve, must pin the
// reference cell so a closed-domain (all-Neumann) pressure system has a unique solution. This
// mirrors test_setReference.cpp's construction but resolves the reference cell from the PIMPLE
// subdict (neoPimpleFoam reads PIMPLE, not PISO) and gates on the pinned cell holding
// pRefValue after the solve. setup_pimple is the all-Neumann lid-driven cavity (p.needReference()
// == true). The rank-0 guard in PDESolver::setReference is PRESERVED (not refactored).
TEST_CASE("neoPimpleFoam closed-domain setReference pin", "[pimpleControl][setref]")
{
    Foam::Time& runTime = *timePtr;
    // Serial executor: the pin assertion is executor-independent and a single MeshAdapter keeps
    // the integration solve cheap and avoids a second-MeshAdapter teardown path.
    auto exec = NeoN::Executor(NeoN::SerialExecutor {});

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    rt.fvSchemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

    // All-Neumann pressure fixture -> needReference() is true (closed domain).
    auto ofU = nf::randomVectorField(runTime, mesh, "U");
    auto ofp = nf::randomScalarField(runTime, mesh, "p");
    ofp.correctBoundaryConditions();
    ofU.correctBoundaryConditions();

    auto& vectorCollection =
        NeoN::finiteVolume::cellCentred::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfP = nf::constructAndRegister(vectorCollection, rt, ofp, false);

    // Tighten the p solver BEFORE mapping so the (soft) reference pin converges to pRefValue:
    // setReference doubles the pinned cell's diagonal and adds diag*pRefValue to its rhs, so the
    // pinned cell only equals pRefValue once the iterative solve has driven the residual down.
    // The fixture's stock p solver (relTol 1e-5) leaves a ~1e-6 floor at the pinned cell — too
    // coarse for a 1e-12 margin. A tight relTol + large maxIter drives the pin to pRefValue.
    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    auto& pDict = solverDict.subDict("p");
    pDict.insert("relTol", NeoN::scalar(1e-12));
    pDict.insert("tolerance", NeoN::scalar(0.0));
    pDict.insert("maxIter", NeoN::label(5000));
    solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

    SECTION("pinned cell holds pRefValue after a closed-domain solve")
    {
        REQUIRE(ofp.needReference());

        // Uniform U=(1,0,0) -> divergence-free phi on the Cartesian fixture, compatible with the
        // all-Neumann pressure system (mirrors test_setReference / [finaliterselect]).
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
        auto nfPhi = nf::constructFrom(rt.exec, rt.nfMesh, ofPhi);

        // Random positive laplacian coefficient; the field name must match setup_pimple's
        // laplacianSchemes entry "laplacian(rAUfNF,p)".
        auto forAUf = nf::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUfNF");
        auto nfrAUf = nf::constructFrom(rt.exec, rt.nfMesh, forAUf);

        // Resolve the reference cell from the PIMPLE subdict. setRefCell makes
        // pRefCell >= 0 only on the rank owning the global ref cell (rank 0 in serial).
        Foam::label pRefCell = 0;
        Foam::scalar pRefValue = 0.0;
        Foam::setRefCell(ofp, ofp.mesh().solutionDict().subDict("PIMPLE"), pRefCell, pRefValue);

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

        auto [numIter, initResNorm, finalResNorm, solveTime] = stats.entries[0];
        REQUIRE(numIter != 0);
        REQUIRE(initResNorm != 0);
        REQUIRE(finalResNorm < initResNorm);

        // Gate: the pinned cell holds pRefValue after the solve.
        auto pHost = nfP.internalVector().copyToHost();
        REQUIRE(pHost.view()[pRefCell] == Catch::Approx(pRefValue).margin(1e-12));
    }
}
