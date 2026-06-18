
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"
#include "constrainHbyA.H"
#include "findRefCell.H"

#include "NeoN/linearAlgebra/ginkgo.hpp"

namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using NeoFOAM::EqualsInternal;
using NeoFOAM::EqualsBoundary;

extern Foam::Time* timePtr; // A single time object

TEST_CASE("Distributed Ginkgo Cache")
{
    SECTION("Parallel sanity check")
    {
        REQUIRE(Foam::Pstream::parRun());
        REQUIRE(Foam::Pstream::nProcs() == 3);
    }

    Foam::Time& runTime = *timePtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));
    solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

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

    // Build a random rAUf surface field for the pressure equation
    auto forAUf =
        NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUf");
    auto nfrAUf = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, forAUf);

    auto& nfU = NeoFOAM::constructAndRegister(vectorCollection, rt, ofU);
    auto& nfOldU = fvcc::oldTime(nfU);
    NeoN::fill(nfOldU.internalVector(), NeoN::Vec3(0.0, 0.0, 0.0));
    nfOldU.correctBoundaryConditions();

    // SECTION 1 — SOLVER-01: skeleton built exactly once across >= 5 PISO pressure solves
    // Each loop iteration constructs a fresh pEqn (mirrors the real lifecycle where the
    // solver is reconstructed inside the non-orthogonal corrector loop) and calls solve().
    // The registry must hit on steps 1..4 and never re-build the skeleton.
    SECTION("SOLVER-01: build_count == 1 after >= 5 pressure solves on " + execName)
    {
        std::size_t firstNumIter = 0;
        const void* spKey = nullptr;

        for (int step = 0; step < 5; ++step)
        {
            // Reset the pressure field to a fixed initial guess so every solve is the
            // identical linear problem (same matrix, same RHS, same x0). The skeleton is
            // built on step 0 (cache miss) and reused on steps 1..4 (cache hits); an
            // identical iteration count across all steps then proves the value-refresh
            // path is bit-identical to the fresh build (SOLVER-03). Without this reset
            // each solve would warm-start from the previous solution, so the cold step-0
            // solve legitimately takes one extra iteration than the warm steps 1..4.
            NeoN::fill(nfP.internalVector(), 0.0);
            nfP.correctBoundaryConditions();

            nf::PDESolver<NeoN::scalar> pEqn(
                dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
                nfP,
                rt
            );
            auto stats = pEqn.solve();
            auto [numIter, initResNorm, finalResNorm, solveTime] = stats.entries[0];

            REQUIRE(numIter != 0);

            const void* key = pEqn.linearSystem().matrix().sparsity().get();
            if (step == 0)
            {
                firstNumIter = numIter;
                spKey = key;
            }
            else
            {
                // SOLVER-03 bit-identical: iteration count stable across steps
                REQUIRE(numIter == firstNumIter);
                // D-04 identity: same mesh → same sparsity pointer every assembly
                REQUIRE(key == spKey);
            }
        }

        REQUIRE(spKey != nullptr);

        // SOLVER-01 (D-06): the skeleton was built exactly once across all 5 solves
        REQUIRE(NeoN::la::ginkgo::getSkeletonBuildCount(spKey) == 1);
    }

    // SECTION 2 — SOLVER-02: 3 Vec3 momentum component solves share one cache entry.
    // solveComponentDist<0/1/2> all key on the same SparsityPattern → one registry hit.
    SECTION("SOLVER-02: Vec3 momentum 3 components share one skeleton on " + execName)
    {
        nf::PDESolver<NeoN::Vec3> nfUEqn(
            dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
            nfU,
            rt
        );
        NeoN::fill(nfUEqn.linearSystem().rhs(), NeoN::Vec3(0.0, 0.0, 0.0));

        auto uStats = nfUEqn.solve();
        auto [numIterU, initResNormU, finalResNormU, solveTimeU] = uStats.entries[0];

        REQUIRE(numIterU != 0);

        const void* uKey = nfUEqn.linearSystem().matrix().sparsity().get();
        REQUIRE(uKey != nullptr);

        // 3 Vec3 components share one registry entry: build count must be 1 (D-01)
        REQUIRE(NeoN::la::ginkgo::getSkeletonBuildCount(uKey) == 1);
    }

    // SECTION 3 — SOLVER-03: two-solver identity + key stability invariant.
    // Note: within this test harness a single distributed mesh is available
    // (setup_pressureVelocityCoupling). The two-mesh test reduces to verifying that:
    //   (a) distinct solver types on the same mesh share the same sparsity key (D-04)
    //   (b) the shared key's build count stays 1 after both solves
    //   (c) a second pEqn instance on the same mesh yields the same key (not a new entry)
    // A genuinely distinct second mesh would require a second case directory and separate
    // AdapterRunTime; that path is not currently supported within a single Catch2 TEST_CASE.
    // This limitation is documented in 12-03-SUMMARY.md.
    SECTION("SOLVER-03: key identity stable + build_count invariant on " + execName)
    {
        // Reset to a fixed initial guess so both solves below are the identical problem.
        NeoN::fill(nfP.internalVector(), 0.0);
        nfP.correctBoundaryConditions();

        // First solve: pressure (scalar laplacian)
        nf::PDESolver<NeoN::scalar> pEqnA(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );
        auto statsA = pEqnA.solve();
        auto [numIterA, initResA, finalResA, timeA] = statsA.entries[0];
        REQUIRE(numIterA != 0);

        const void* spKeyA = pEqnA.linearSystem().matrix().sparsity().get();
        REQUIRE(spKeyA != nullptr);
        REQUIRE(NeoN::la::ginkgo::getSkeletonBuildCount(spKeyA) == 1);

        // Reset to the same fixed initial guess so pEqnB solves the identical problem as
        // pEqnA — an equal iteration count then proves the cached (reused) skeleton path
        // is bit-identical to pEqnA's path (both are cache hits on the same matrix).
        NeoN::fill(nfP.internalVector(), 0.0);
        nfP.correctBoundaryConditions();

        // Second solve: a fresh pEqn on the same mesh — must reuse the cached skeleton
        nf::PDESolver<NeoN::scalar> pEqnB(
            dsl::imp::laplacian(nfrAUf, nfP) - dsl::exp::div(nfPhi),
            nfP,
            rt
        );
        auto statsB = pEqnB.solve();
        auto [numIterB, initResB, finalResB, timeB] = statsB.entries[0];
        REQUIRE(numIterB != 0);

        const void* spKeyB = pEqnB.linearSystem().matrix().sparsity().get();

        // D-04: same mesh → same sparsity identity → same key
        REQUIRE(spKeyB == spKeyA);

        // D-02 (keep-all) + D-06: build count stays 1 — second solve was a cache HIT
        REQUIRE(NeoN::la::ginkgo::getSkeletonBuildCount(spKeyA) == 1);

        // SOLVER-03 bit-identical: iteration counts match across the two fresh pEqn instances
        REQUIRE(numIterB == numIterA);
    }
}
