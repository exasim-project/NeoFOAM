// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Test for applyVelocityDamping (the port of OpenFOAM's fv::velocityDampingConstraint) and the
// PDE::setPostRelaxConstraint hook that applies it. The properties under test are those of
// upstream velocityDampingConstraint::addDamping: only cells with |U| > UMax are touched, only
// their diagonal changes, by exactly C * V^(2/3) * (|U| - UMax), and the rhs is left alone.

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "NeoFOAM/algorithms/velocityDamping.hpp"

#include <cmath>
#include <vector>

namespace dsl = NeoN::dsl;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

namespace
{

// Host-side diagonal index of every row, found by scanning the CSR column indices.
template<typename LinearSystem>
std::vector<NeoN::localIdx> hostDiagIdx(const LinearSystem& ls)
{
    auto rowOffs = ls.matrix().rowOffs().copyToHost();
    auto colIdxs = ls.matrix().colIdxs().copyToHost();
    const auto rowOffsV = rowOffs.view();
    const auto colIdxsV = colIdxs.view();
    std::vector<NeoN::localIdx> diag(rowOffsV.size() - 1, -1);
    for (std::size_t row = 0; row + 1 < rowOffsV.size(); ++row)
    {
        for (auto idx = rowOffsV[row]; idx < rowOffsV[row + 1]; ++idx)
        {
            if (static_cast<std::size_t>(colIdxsV[static_cast<std::size_t>(idx)]) == row)
            {
                diag[row] = idx;
            }
        }
    }
    return diag;
}

} // namespace

TEST_CASE("VelocityDamping")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

    // Random U has |U| <= sqrt(3); push a few scattered cells well above UMax so the damped set
    // is known exactly.
    const NeoN::scalar UMax = 5.0;
    const NeoN::scalar dampingC = 2.0;
    std::vector<NeoN::localIdx> fast {0, 3, 7};
    auto ofU = randomVectorField(runTime, mesh, "U");
    for (std::size_t i = 0; i < fast.size(); ++i)
    {
        if (fast[i] < mesh.nCells())
        {
            ofU.primitiveFieldRef()[fast[i]] =
                Foam::vector(10.0 + static_cast<Foam::scalar>(i), -3.0, 1.0);
        }
    }
    ofU.correctBoundaryConditions();

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

    auto& vectorCollection =
        NeoN::finiteVolume::cellCentred::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfU = NeoFOAM::constructAndRegister(vectorCollection, rt, ofU, false);
    auto nfNu = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofNu);

    const auto nCells = rt.nfMesh.nCells();
    std::vector<NeoN::scalar> expectedBoost(static_cast<std::size_t>(nCells), 0.0);
    NeoN::label expectedDamped = 0;
    for (Foam::label celli = 0; celli < mesh.nCells(); ++celli)
    {
        const auto magU = Foam::mag(ofU.primitiveField()[celli]);
        if (magU > UMax)
        {
            const auto cbrtV = std::cbrt(mesh.V()[celli]);
            expectedBoost[static_cast<std::size_t>(celli)] =
                dampingC * cbrtV * cbrtV * (magU - UMax);
            ++expectedDamped;
        }
    }
    REQUIRE(expectedDamped > 0);

    SECTION("only the diagonal of cells above UMax is boosted on " + execName)
    {
        nf::PDE<NeoN::Vec3> UEqn(dsl::imp::laplacian(nfNu, nfU), nfU, rt);

        // Reference: the same system without the constraint. Copy to the host, since the next
        // assembly reuses the same storage.
        auto& refLs = UEqn.assembleAndRelax();
        auto refValues = refLs.matrix().values().copyToHost();
        auto refRhs = refLs.rhs().copyToHost();

        NeoN::label nDamped = -1;
        UEqn.setPostRelaxConstraint([&](auto& ls)
                                    { nDamped = nf::applyVelocityDamping(ls, nfU, UMax, dampingC); }
        );
        auto& ls = UEqn.assembleAndRelax();

        REQUIRE(nDamped == expectedDamped);

        auto values = ls.matrix().values().copyToHost();
        auto rhs = ls.rhs().copyToHost();
        const auto diag = hostDiagIdx(ls);

        std::vector<NeoN::scalar> expected(refValues.view().begin(), refValues.view().end());
        for (std::size_t celli = 0; celli < expectedBoost.size(); ++celli)
        {
            expected[static_cast<std::size_t>(diag[celli])] += expectedBoost[celli];
        }
        for (std::size_t idx = 0; idx < expected.size(); ++idx)
        {
            REQUIRE(values.view()[idx] == Catch::Approx(expected[idx]).epsilon(1e-12));
        }
        for (std::size_t celli = 0; celli < static_cast<std::size_t>(nCells); ++celli)
        {
            REQUIRE(rhs.view()[celli][0] == refRhs.view()[celli][0]);
            REQUIRE(rhs.view()[celli][1] == refRhs.view()[celli][1]);
            REQUIRE(rhs.view()[celli][2] == refRhs.view()[celli][2]);
        }
    }

    SECTION("the constraint also runs on the solve(rhs) path on " + execName)
    {
        nf::PDE<NeoN::Vec3> UEqn(dsl::imp::laplacian(nfNu, nfU), nfU, rt);

        NeoN::label nDamped = -1;
        UEqn.setPostRelaxConstraint([&](auto& ls)
                                    { nDamped = nf::applyVelocityDamping(ls, nfU, UMax, dampingC); }
        );
        auto ofp = randomScalarField(runTime, mesh, "p");
        ofp.correctBoundaryConditions();
        auto& nfP = NeoFOAM::constructAndRegister(vectorCollection, rt, ofp, false);

        // Damping is evaluated on U as it is before this solve modifies it.
        UEqn.solve(-1.0 * dsl::exp::grad(nfP));

        REQUIRE(nDamped == expectedDamped);
    }
}
