// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/pdeSolver.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"

namespace dsl = NeoN::dsl;

namespace NeoFOAM
{

/**
 * @brief Caches a NeoN::la::Solver instance for a given field to avoid reconstructing
 *        it (and re-parsing the solver dictionary) on every time-step iteration.
 *
 * Usage:
 * @code
 *   auto uSolver = nf::Solver(U, rt);   // once before the time loop
 *   // inside the loop:
 *   nf::PDESolver<NeoN::Vec3> UEqn(dsl::imp::ddt(U) + ...);
 *   uSolver.solve(UEqn, -1.0 * dsl::exp::grad(p));   // momentum predictor
 *   uSolver.assemble(UEqn);                           // or explicit assembly only
 * @endcode
 */
template<
    typename ValueType,
    typename MatrixValueType = NeoN::scalar,
    typename IndexType = NeoN::localIdx>
class Solver
{
    using VolumeField = NeoN::finiteVolume::cellCentred::VolumeField<ValueType>;
    using PDESolverType = PDESolver<ValueType, MatrixValueType, IndexType>;
    using LinearSystemType = NeoN::la::LinearSystem<MatrixValueType, ValueType>;

public:

    Solver(VolumeField& field, RunTime& rt)
        : field_(field)
        , rt_(rt)
        , cachedSolver_(createCachedSolver(field, rt))
    {}

    NeoN::la::SolverStats solve(PDESolverType& pde)
    {
        return pde.solveWith(cachedSolver_, field_, rt_);
    }

    NeoN::la::SolverStats solve(PDESolverType& pde, dsl::SpatialOperator<ValueType>&& rhs)
    {
        return pde.solveWith(cachedSolver_, field_, rt_, std::move(rhs));
    }

    LinearSystemType& assemble(PDESolverType& pde) { return pde.assemble(field_, rt_); }

    LinearSystemType& assembleAndRelax(PDESolverType& pde)
    {
        pde.assemble(field_, rt_);
        pde.relaxOwnedLs();
        return pde.linearSystem();
    }

private:

    static NeoN::la::Solver createCachedSolver(VolumeField& field, RunTime& rt)
    {
        auto solverDict = rt.fvSolutionDict.subDict("solvers").subDict(field.name);
        PDESolverType::stripNeoFOAMKeys(solverDict);
        return NeoN::la::Solver(field.exec(), solverDict);
    }

    VolumeField& field_;
    RunTime& rt_;
    NeoN::la::Solver cachedSolver_;
};

// Deduction guide: Solver(field, rt) deduces ValueType from the VolumeField argument.
template<typename ValueType>
Solver(NeoN::finiteVolume::cellCentred::VolumeField<ValueType>&, RunTime&) -> Solver<ValueType>;

} // namespace NeoFOAM
