// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Dedicated test for PDE::setConstraints, the hard cell pin behind the epsilon/omega wall
// functions. The property under test is the one fvMatrix::setValuesFromList provides by
// assigning `psi[celli] = value` before it touches the matrix: a constrained cell holds its
// value no matter what the linear solver subsequently does. Pinning only the matrix leaves that
// cell to the solver, and a solve that exits without iterating leaves it at its previous value.

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

namespace dsl = NeoN::dsl;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

TEST_CASE("PDESetConstraints")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);

    auto ofp = NeoFOAM::randomScalarField(runTime, mesh, "p");
    ofp.correctBoundaryConditions();

    auto& vectorCollection =
        NeoN::finiteVolume::cellCentred::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfP = NeoFOAM::constructAndRegister(vectorCollection, rt, ofp, false);

    const auto nCells = rt.nfMesh.nCells();

    // Pin a handful of scattered cells to values far from anything the random field holds, so a
    // cell that kept its pre-solve value cannot pass by coincidence.
    // Build the pin arrays on the host, then hand them to the executor at construction.
    // Filling a copyToHost() result and assigning it back would abort on "Executors are not
    // the same" for every non-serial executor.
    std::vector<NeoN::localIdx> pinned {0, 3, 7};
    std::vector<NeoN::scalar> maskHost(static_cast<std::size_t>(nCells), NeoN::scalar(0));
    std::vector<NeoN::scalar> valueHost(static_cast<std::size_t>(nCells), NeoN::scalar(0));
    for (std::size_t i = 0; i < pinned.size(); ++i)
    {
        if (pinned[i] >= nCells) continue;
        const auto cell = static_cast<std::size_t>(pinned[i]);
        maskHost[cell] = NeoN::scalar(1);
        valueHost[cell] = NeoN::scalar(100) + static_cast<NeoN::scalar>(i);
    }
    NeoN::Vector<NeoN::scalar> mask(exec, maskHost);
    NeoN::Vector<NeoN::scalar> values(exec, valueHost);

    // maxIter 0 -> the Ginkgo iteration criterion is met immediately, so the solver returns
    // without touching the solution. This is the deterministic stand-in for the "No Iterations 0"
    // solves a transient throws up, where the pin is otherwise silently skipped.
    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    NeoN::Dictionary pDict = solverDict.subDict("p");
    pDict.insert("maxIter", NeoN::label(0));
    solverDict.subDict("p") = nf::mapFvSolution(pDict);

    SECTION("a constrained cell holds its value through a zero-iteration solve on " + execName)
    {
        auto forAUf =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, {0, 0, 1, 0, 0}, "rAUfNF");
        auto nfrAUf = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, forAUf);

        nf::PDE<NeoN::scalar> pEqn(dsl::imp::laplacian(nfrAUf, nfP), nfP, rt);
        pEqn.setConstraints(mask, values);

        auto stats = pEqn.solve();

        // Guard the premise: if the solver did iterate, the assertion below would no longer be
        // testing the assignment.
        auto [numIter, initResNorm, finalResNorm, solveTime] = stats.entries[0];
        REQUIRE(numIter == 0);

        auto psiHost = nfP.internalVector().copyToHost();
        for (auto cell : pinned)
        {
            if (cell >= nCells) continue;
            REQUIRE(psiHost.view()[cell] == valueHost[static_cast<std::size_t>(cell)]);
        }
    }
}
