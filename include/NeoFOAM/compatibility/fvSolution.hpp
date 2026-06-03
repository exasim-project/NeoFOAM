// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
/* This file implements comparison operator to compare OpenFOAM and corresponding NeoFOAM fields
 * TODO the comparison operator only make sense for testing purposes
 * so this should be part of the tests
 */
#pragma once

#include "NeoN/core/dictionary.hpp"


namespace NeoFOAM
{

void updateSolver(NeoN::Dictionary& solverDict);

void updatePreconditioner(NeoN::Dictionary& solverDict);

/* @brief Map an OpenFOAM fvSolution sub-dictionary to NeoN/Ginkgo settings.
 *
 * The returned dictionary additionally carries a "reportName" meta key holding
 * the Ginkgo solver/preconditioner label (e.g. "Ic+Cg") used by the per-solve
 * residual report. The Ginkgo backend's config parser ignores that key.
 *
 * @param solverDict  The OpenFOAM solver sub-dictionary (e.g. solvers/p).
 */
NeoN::Dictionary mapFvSolution(const NeoN::Dictionary& solverDict);

} // namespace NeoFOAM
