// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2026 NeoFOAM authors
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/label.hpp"

namespace NeoFOAM
{

// Per-field residual the outer loop is fed each pass: {initResNorm, finalResNorm}.
// The native control mirrors OpenFOAM pimpleControl::criteriaSatisfied(): the
// absolute criterion compares the FINAL residual, the relative criterion compares
// finalResNorm / storedInitial. Feeding both makes both checks available without
// a Foam::solverPerformance / solverPerformanceDict round-trip.
using FieldResidual = std::pair<NeoN::scalar, NeoN::scalar>;
using ResidualMap = std::map<std::string, FieldResidual>;

/* @brief NeoFOAM-native outer PIMPLE loop control (does NOT reuse OpenFOAM's pimpleControl).
 *
 * Owns the corr_ / converged_ state machine that drives the outer corrector loop. It mirrors
 * OpenFOAM pimpleControl::loop() (pimpleControl.C:200-265) and criteriaSatisfied()
 * (pimpleControl.C:61-126) byte-for-byte on the case side: it reads the SAME
 * fvSolution.PIMPLE subdict a stock pimpleFoam case uses (nOuterCorrectors,
 * residualControl{ p{ tolerance; relTol; } U{ ... } }) and applies per-field abs-OR-rel
 * convergence with no check on the first outer iteration and a two-phase converged_ exit
 * (an extra final pass after the criteria are met, so *Final factors/tolerances fire).
 *
 * Unlike OpenFOAM's pimpleControl it is FED NeoN scalar residuals directly via loop(residuals) —
 * there is no mesh_.data().solverPerformanceDict() round-trip (rationale: the loop is not
 * a hot path, so "performant" means lean / decoupled / portable).
 */
class PimpleControl
{
public:

    // Reads the PIMPLE subdict from fvSolution (nOuterCorrectors, residualControl{}).
    // A missing / malformed PIMPLE block degrades to nOuterCorrectors == 1 and an empty
    // residualControl (OpenFOAM defaults) — never throws.
    explicit PimpleControl(const NeoN::Dictionary& fvSolution);

    // True on the last outer pass: converged_ || (corr_ == nOuterCorr_). Drives *Final
    // relaxation factors + <field>Final solver subdict selection (pimpleControlI.H:92-95).
    [[nodiscard]] bool finalIter() const;

    // True on the first outer pass (corr_ == 1) — convergence is never checked here.
    [[nodiscard]] bool firstIter() const;

    // Call once per outer pass. `residuals` holds the {init,final} residuals from the PREVIOUS
    // pass. Returns true while another outer corrector should run; false when the loop is done
    // (count exhausted OR residual criteria satisfied after the extra final pass).
    bool loop(const ResidualMap& residuals);

    [[nodiscard]] NeoN::label nOuterCorrectors() const { return nOuterCorr_; }

private:

    bool criteriaSatisfied(const ResidualMap& residuals);

    struct FieldCtrl
    {
        std::string name;
        NeoN::scalar absTol;
        NeoN::scalar relTol;
        NeoN::scalar initialResidual;
    };

    NeoN::label nOuterCorr_ = 1;
    NeoN::label corr_ = 0;
    bool converged_ = false;
    std::vector<FieldCtrl> residualControl_;
};

} // namespace NeoFOAM
