// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/solutionControl/pimpleControl.hpp"

namespace NeoFOAM
{

// Coerce an int-typed dictionary entry to scalar; bare-integer tolerance values
// (e.g. `tolerance 0;`) are stored as int and rejected by get<scalar>.
static NeoN::scalar asScalar(const NeoN::Dictionary& d, const std::string& k)
{
    return d.isType<int>(k) ? NeoN::scalar(d.get<int>(k)) : d.get<NeoN::scalar>(k);
}

PimpleControl::PimpleControl(const NeoN::Dictionary& fvSolution)
{
    if (!fvSolution.contains("PIMPLE") || !fvSolution.isDict("PIMPLE"))
    {
        return;
    }

    const auto& pimple = fvSolution.subDict("PIMPLE");

    if (pimple.contains("nOuterCorrectors"))
    {
        nOuterCorr_ = pimple.get<NeoN::label>("nOuterCorrectors");
    }

    if (pimple.contains("residualControl") && pimple.isDict("residualControl"))
    {
        const auto& rc = pimple.subDict("residualControl");
        for (const auto& field : rc.keys())
        {
            if (!rc.isDict(field))
            {
                continue;
            }
            const auto& fieldDict = rc.subDict(field);

            const NeoN::scalar absTol =
                fieldDict.contains("tolerance") ? asScalar(fieldDict, "tolerance") : 0.0;
            const NeoN::scalar relTol =
                fieldDict.contains("relTol") ? asScalar(fieldDict, "relTol") : 0.0;

            residualControl_.push_back(FieldCtrl {field, absTol, relTol, 0.0});
        }
    }
}

bool PimpleControl::finalIter() const { return converged_ || (corr_ == nOuterCorr_); }

bool PimpleControl::firstIter() const { return corr_ == 1; }

bool PimpleControl::loop(const ResidualMap& residuals)
{
    // Two-phase convergence exit: set converged_ and run one extra final pass so
    // *Final factors fire, then exit on the subsequent call.
    ++corr_;

    if (corr_ == nOuterCorr_ + 1)
    {
        corr_ = 0;
        return false;
    }

    if (converged_ || criteriaSatisfied(residuals))
    {
        if (converged_)
        {
            corr_ = 0;
            converged_ = false;
            return false;
        }
        converged_ = true;
    }

    return true;
}

bool PimpleControl::criteriaSatisfied(const ResidualMap& residuals)
{
    if (corr_ == 1 || residualControl_.empty() || finalIter())
    {
        return false;
    }

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
        const NeoN::scalar ini = it->second.first;
        const NeoN::scalar fin = it->second.second;

        checked = true;

        const bool absCheck = fin < fc.absTol;

        if (storeIni)
        {
            fc.initialResidual = ini;
        }
        // +1e-37 guards against division by zero (mirrors OpenFOAM's ROOTVSMALL).
        const NeoN::scalar rel = storeIni ? 1.0 : fin / (fc.initialResidual + 1e-37);
        const bool relCheck = !storeIni && rel < fc.relTol;

        achieved = achieved && (absCheck || relCheck);
    }

    return checked && achieved;
}

} // namespace NeoFOAM
