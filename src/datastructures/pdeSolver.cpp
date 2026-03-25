// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/datastructures/pdeSolver.hpp"

namespace NeoFOAM
{


template<>
NeoN::la::LinearSystem<NeoN::Vec3>
PDESolver<NeoN::Vec3, NeoN::localIdx>::assemble(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
{
    auto rhsExpr = dsl::Expression<NeoN::Vec3>(-1.0 * rhs);
    rhsExpr.read(runTime_.fvSchemesDict);
    auto ls = NeoN::la::LinearSystem<NeoN::Vec3>(assemble());

    auto expTmp = rhsExpr.explicitOperation(psi_.mesh().nCells());

    auto [vol, expSource, rhsV] = NeoN::views(psi_.mesh().cellVolumes(), expTmp, ls.rhs());
    NeoN::parallelFor(
        psi_.exec(),
        {0, rhsV.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rhsV[i] -= expSource[i] * vol[i]; }
    );

    return ls;
}

template<>
NeoN::la::LinearSystem<NeoN::scalar>
PDESolver<NeoN::scalar, NeoN::localIdx>::assemble(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
{}

template<typename ValueType, typename IndexType>
NeoN::la::LinearSystem<ValueType>
PDESolver<ValueType, IndexType>::assemble(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
{
    auto rhsExpr = dsl::Expression<ValueType>(-1.0 * rhs);
    rhsExpr.read(runTime_.fvSchemesDict);
    auto ls = NeoN::la::LinearSystem<ValueType>(assemble());

    auto expTmp = rhsExpr.explicitOperation(psi_.mesh().nCells());

    auto [vol, expSource, rhsV] = NeoN::views(psi_.mesh().cellVolumes(), expTmp, ls.rhs());
    NeoN::parallelFor(
        psi_.exec(),
        {0, rhsV.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rhsV[i] -= expSource[i] * vol[i]; }
    );

    return ls;
}

template<typename ValueType, typename IndexType>
NeoN::la::SolverStats PDESolver<ValueType, IndexType>::solve(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
{
    auto ls = assemble(std::move(rhs));

    auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
    auto fvSolution = solverDict.subDict(psi_.name);
    auto solver = NeoN::la::Solver(psi_.exec(), fvSolution);

    // Do some sanity checks before trying to solve
    // NF_ASSERT(ls.exec() == solution.exec(), "Executors are not the same");
    return solver.solve(ls, psi_.internalVector());
}

// instantiate the template class
template class PDESolver<NeoN::scalar, NeoN::localIdx>;
template class PDESolver<NeoN::Vec3, NeoN::localIdx>;

}
