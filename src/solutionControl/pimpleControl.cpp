// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/solutionControl/pimpleControl.hpp"

namespace NeoFOAM
{

// Int-tolerant scalar read copied from lookupRelaxation (fvSolution.cpp):
// OpenFOAM writes whole-number residual tolerances as bare integers (e.g. `tolerance 0;`),
// which `get<scalar>` rejects with bad_any_cast (no int->scalar coercion). Coerce an
// int-typed entry to scalar, otherwise read it as scalar.
static NeoN::scalar asScalar(const NeoN::Dictionary& d, const std::string& k)
{
    return d.isType<int>(k) ? NeoN::scalar(d.get<int>(k)) : d.get<NeoN::scalar>(k);
}

PimpleControl::PimpleControl(const NeoN::Dictionary& fvSolution)
{
    // isDict guard: a missing or malformed PIMPLE block degrades to the OpenFOAM
    // defaults (nOuterCorrectors == 1, empty residualControl) instead of throwing.
    if (!fvSolution.contains("PIMPLE") || !fvSolution.isDict("PIMPLE"))
    {
        return;
    }

    const auto& pimple = fvSolution.subDict("PIMPLE");

    // nOuterCorrectors is a Foam::label -> NeoN::label. Read with get<NeoN::label>, NEVER
    // get<scalar> (bare-int label values would bad_any_cast through get<scalar>).
    if (pimple.contains("nOuterCorrectors"))
    {
        nOuterCorr_ = pimple.get<NeoN::label>("nOuterCorrectors");
    }

    // residualControl is an optional sub-dict of per-field { tolerance; relTol; } entries.
    if (pimple.contains("residualControl") && pimple.isDict("residualControl"))
    {
        const auto& rc = pimple.subDict("residualControl");
        for (const auto& field : rc.keys())
        {
            // Each residual-controlled field is itself a sub-dict (p { ... } / U { ... }).
            if (!rc.isDict(field))
            {
                continue;
            }
            const auto& fieldDict = rc.subDict(field);

            // Guard each sub-read with contains(); default a missing key to 0.0. The
            // int-tolerant asScalar mirrors the lookupRelaxation idiom.
            const NeoN::scalar absTol =
                fieldDict.contains("tolerance") ? asScalar(fieldDict, "tolerance") : 0.0;
            const NeoN::scalar relTol =
                fieldDict.contains("relTol") ? asScalar(fieldDict, "relTol") : 0.0;

            residualControl_.push_back(FieldCtrl {field, absTol, relTol, 0.0});
        }
    }
}

bool PimpleControl::finalIter() const
{
    // pimpleControlI.H:92-95.
    return converged_ || (corr_ == nOuterCorr_);
}

bool PimpleControl::firstIter() const { return corr_ == 1; }

bool PimpleControl::loop(const ResidualMap& residuals)
{
    // Mirror pimpleControl::loop() (pimpleControl.C:200-265) — two-phase converged
    // exit: when the criteria are met we set converged_ = true and run ONE MORE pass (on which
    // finalIter() is true so *Final factors/tolerances fire), then exit AFTER that pass.
    ++corr_;

    if (corr_ == nOuterCorr_ + 1)
    {
        corr_ = 0; // ran all correctors -> stop + reset for the next time step
        return false;
    }

    if (converged_ || criteriaSatisfied(residuals))
    {
        if (converged_)
        {
            corr_ = 0;
            converged_ = false;
            return false; // exit AFTER the extra final pass
        }
        converged_ = true; // run ONE MORE pass
    }

    return true;
}

bool PimpleControl::criteriaSatisfied(const ResidualMap& residuals)
{
    // Mirror pimpleControl::criteriaSatisfied() (pimpleControl.C:61-126); FED residuals
    // directly (no mesh_.data().solverPerformanceDict() round-trip).

    // No checks on the first iteration — nothing has converged yet. Also skip when no
    // residualControl is configured or on the forced final iteration.
    if (corr_ == 1 || residualControl_.empty() || finalIter())
    {
        return false;
    }

    // Store the initial residual on the 2nd PIMPLE iteration (pimpleControlI.H:79-83).
    const bool storeIni = (corr_ == 2);

    bool achieved = true;
    bool checked = false; // safety that some checks were indeed performed
    for (auto& fc : residualControl_)
    {
        auto it = residuals.find(fc.name);
        if (it == residuals.end())
        {
            continue;
        }
        const NeoN::scalar ini = it->second.first;  // initResNorm
        const NeoN::scalar fin = it->second.second; // finalResNorm

        checked = true;

        // OpenFOAM: the absolute criterion compares the FINAL residual (A1).
        const bool absCheck = fin < fc.absTol;

        if (storeIni)
        {
            fc.initialResidual = ini;
        }
        // Relative criterion compares finalResNorm / storedInitial; not evaluated on the
        // store-initial pass. The +1e-37 mirrors OpenFOAM's ROOTVSMALL guard.
        const NeoN::scalar rel = storeIni ? 1.0 : fin / (fc.initialResidual + 1e-37);
        const bool relCheck = !storeIni && rel < fc.relTol;

        achieved = achieved && (absCheck || relCheck); // per-field abs-OR-rel
    }

    return checked && achieved;
}

} // namespace NeoFOAM
