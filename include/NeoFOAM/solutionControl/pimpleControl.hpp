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

// Per-field residual fed to the outer loop each pass: {initResNorm, finalResNorm}.
using FieldResidual = std::pair<NeoN::scalar, NeoN::scalar>;
using ResidualMap = std::map<std::string, FieldResidual>;

/** @brief NeoFOAM-native outer PIMPLE loop control.
 *
 * Reads nOuterCorrectors and residualControl from the fvSolution PIMPLE subdict.
 * Applies per-field absolute-OR-relative convergence; skips the check on the first
 * iteration; runs an extra final pass after convergence so *Final factors fire.
 * Residuals are fed directly via loop(residuals) — no OpenFOAM round-trip.
 */
class PimpleControl
{
public:

    // A missing PIMPLE subdict degrades to nOuterCorrectors == 1 with no residual control.
    explicit PimpleControl(const NeoN::Dictionary& fvSolution);

    // True on the last outer pass, enabling *Final relaxation factors and solver subdicts.
    [[nodiscard]] bool finalIter() const;

    [[nodiscard]] bool firstIter() const;

    // Returns true while another outer corrector should run.
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
