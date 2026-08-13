// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
/* This file implements comparison operator to compare OpenFOAM and corresponding NeoFOAM fields
 * TODO the comparison operator only make sense for testing purposes
 * so this should be part of the tests
 */
#pragma once

#include <optional>
#include <string>

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/primitives/scalar.hpp"

#include "NeoFOAM/datastructures/runTime.hpp"


namespace NeoFOAM
{

void updateSolver(NeoN::Dictionary& solverDict);

void updatePreconditioner(NeoN::Dictionary& solverDict);

/* @brief Which NeoN::la system matrix format a solver sub-dict selects.
 *
 * Picks which PDE<..., SystemMatrixType>/Solver<..., SystemMatrixType> template instantiation
 * a solver equation (e.g. neoIcoFoam's U or p) should run with. This is a compile-time C++
 * template parameter, not a Ginkgo setting -- the binary must already have compiled in the
 * instantiation this selects. See matrixFormat() below for the dictionary entry that picks
 * between them at startup, independently per equation.
 */
enum class MatrixFormat
{
    CSR,
    ELL
};

/* @brief Look up which matrix format (CSR or ELL) a solver sub-dict selects.
 *
 * Parses the "matrixFormat" entry of a per-field solver sub-dict (e.g. solvers.p.matrixFormat,
 * solvers.U.matrixFormat) -- each equation picks its own format independently. Falls back to
 * @p defaultFormat when the key is absent; callers should pass whatever their application's
 * matrix format was before this dictionary entry existed (e.g. neoIcoFoam, which used to
 * hardcode ELL, passes MatrixFormat::ELL as its default so cases without the key keep running
 * exactly as they did before this option existed). There is no project-wide default: an
 * application with no prior hardcoded format should pass MatrixFormat::CSR, matching
 * NeoN::la::PDE/Solver/LinearSystem's own default SystemMatrixType template argument.
 *
 * Only the base "p"/"U" sub-dicts are read for this -- NOT "pFinal"/"UFinal". Matrix format is
 * a compile-time template parameter fixed once for the whole run (see runCase in neoIcoFoam.cpp),
 * unlike tolerance/relTol/preconditioner, which pde.hpp's PDE::assemble genuinely does re-read
 * from "<field>Final" on the last PISO/outer iteration. A "matrixFormat" placed inside a Final
 * sub-dict is therefore inert: it is stripped by PDE::stripNeoFOAMKeys like any other key headed
 * for Ginkgo, but nothing ever reads it back out to select a template.
 *
 * The key must be stripped before the sub-dict reaches Ginkgo's config parser -- see
 * PDE::stripNeoFOAMKeys, which callers (e.g. NeoFOAM::Solver::createCachedSolver) already run.
 *
 * @param solverDict     A per-field solver sub-dictionary (e.g. solvers/p, solvers/U) -- the
 *                        base dict, not its Final variant; see above.
 * @param defaultFormat  Format to use when "matrixFormat" is absent.
 */
MatrixFormat matrixFormat(const NeoN::Dictionary& solverDict, MatrixFormat defaultFormat);

/* @brief Look up the equation under-relaxation factor for a field from fvSolution.
 *
 * Parses the OpenFOAM `relaxationFactors.equations` sub-dictionary and returns the
 * scalar relaxation factor for the given field. When `finalIter` is true the
 * `<field>Final` key is preferred (Final-suffix seam), falling back to the base
 * `<field>` key. Returns std::nullopt when no entry exists -> the caller uses 1.0 ->
 * the kernel is a no-op (matches OpenFOAM's "no entry, no relax" semantics).
 *
 * Keeping this parsing in NeoFOAM means NeoN's applyMatrixRelaxation only ever
 * sees a plain scalar; NeoN stays OpenFOAM-dictionary-agnostic.
 *
 * @param fvSolution  The parsed fvSolution dictionary (e.g. RunTime::fvSolutionDict).
 * @param field       The field name (e.g. "U").
 * @param finalIter   Whether the final outer iteration is active (selects the *Final key).
 */
std::optional<NeoN::scalar>
lookupEqnRelaxation(const NeoN::Dictionary& fvSolution, const std::string& field, bool finalIter);

/* @brief Look up the explicit field under-relaxation factor for a field from fvSolution.
 *
 * Parses the OpenFOAM `relaxationFactors.fields` sub-dictionary and returns the scalar
 * relaxation factor for the given field. When `finalIter` is true the `<field>Final` key is
 * preferred (Final-suffix seam), falling back to the base `<field>` key. Returns
 * std::nullopt when no entry exists -> the caller uses 1.0 -> the kernel is a no-op (matches
 * OpenFOAM's "no entry, no relax" semantics).
 *
 * `relaxationFactors.equations` and `relaxationFactors.fields` are independent dicts:
 * a field present only under `equations` (e.g. momentum `U`) returns nullopt here, so
 * field-URF is a no-op on it and momentum is never double-relaxed. Shares the int-tolerant,
 * isDict-guarded lookup body with lookupEqnRelaxation.
 *
 * @param fvSolution  The parsed fvSolution dictionary (e.g. RunTime::fvSolutionDict).
 * @param field       The field name (e.g. "p").
 * @param finalIter   Whether the final outer iteration is active (selects the *Final key).
 */
std::optional<NeoN::scalar>
lookupFieldRelaxation(const NeoN::Dictionary& fvSolution, const std::string& field, bool finalIter);

/* @brief Map an OpenFOAM fvSolution sub-dictionary to NeoN/Ginkgo settings.
 *
 * The returned dictionary additionally carries a "reportName" meta key holding
 * the Ginkgo solver/preconditioner label (e.g. "Ic+Cg") used by the per-solve
 * residual report. The Ginkgo backend's config parser ignores that key.
 *
 * @param solverDict  The OpenFOAM solver sub-dictionary (e.g. solvers/p).
 */
NeoN::Dictionary mapFvSolution(const NeoN::Dictionary& solverDict);

/**
 * @brief Map the p, U, pFinal, and UFinal solver subdicts in rt.fvSolutionDict to NeoN/Ginkgo
 * format.
 *
 * pFinal and UFinal are skipped when absent (not all cases define Final subdicts).
 */
void createMappedFvSolutionDicts(RunTime& rt);

} // namespace NeoFOAM
