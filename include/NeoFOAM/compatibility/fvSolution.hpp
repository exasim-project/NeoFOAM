// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
/* This file implements comparison operator to compare OpenFOAM and corresponding NeoFOAM fields
 * TODO the comparison operator only make sense for testing purposes
 * so this should be part of the tests
 */
#pragma once

#include <string>

#include "NeoN/core/dictionary.hpp"


namespace NeoFOAM
{

void updateSolver(NeoN::Dictionary& solverDict);

void updatePreconditioner(NeoN::Dictionary& solverDict);

/* @brief Map an OpenFOAM fvSolution sub-dictionary to NeoN/Ginkgo settings.
 *
 * @param solverDict  The OpenFOAM solver sub-dictionary (e.g. solvers/p).
 * @param fieldName   Optional field name, used only for an OpenFOAM-style
 *                    "preconditioner+solver" report logged once at setup
 *                    (e.g. "DICPCG"). No effect on the returned dictionary.
 */
NeoN::Dictionary mapFvSolution(const NeoN::Dictionary& solverDict, const std::string& fieldName = "");

} // namespace NeoFOAM
