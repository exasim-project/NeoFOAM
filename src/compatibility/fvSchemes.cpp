// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2026 NeoFOAM authors


#include "NeoFOAM/compatibility/fvSchemes.hpp"

#include <map>

#include <NeoN/core/logging.hpp>


namespace NeoFOAM
{

void updateDdtSchemes(NeoN::Dictionary& schemeDict)
{
    if (!schemeDict.contains("ddtSchemes"))
    {
        return;
    }

    NeoN::Dictionary& ddtSchemes = schemeDict.subDict("ddtSchemes");

    static const std::map<std::string, std::string> schemeMap = {
        {"Euler", "BDF1"},
        {"backward", "BDF2"},
    };

    for (const auto& key : ddtSchemes.keys())
    {
        if (!ddtSchemes.isType<std::string>(key))
        {
            // Ignore non-string entries (future-proof)
            continue;
        }

        std::string& schemeName = ddtSchemes.get<std::string>(key);
        auto it = schemeMap.find(schemeName);
        if (it != schemeMap.end())
        {
            NeoN::Logging::warn(
                "Replacing ddt scheme '{}' → '{}' for entry '{}'",
                schemeName,
                it->second,
                key
            );
            schemeName = it->second;
        }
    }
}

// OpenFOAM allows "bounded <scheme> ..." as a div scheme prefix for bounded convection.
// NeoN has no separate "bounded" constructor — strip the prefix so NeoN sees "Gauss ...".
static void stripBoundedDivSchemes(NeoN::Dictionary& schemeDict)
{
    if (!schemeDict.contains("divSchemes")) return;
    NeoN::Dictionary& divSchemes = schemeDict.subDict("divSchemes");
    for (const auto& key : divSchemes.keys())
    {
        if (!divSchemes.isType<NeoN::TokenList>(key)) continue;
        NeoN::TokenList& tl = divSchemes.get<NeoN::TokenList>(key);
        if (tl.size() == 0) continue;
        try
        {
            if (tl.get<std::string>(0) == "bounded")
            {
                NeoN::Logging::warn("Stripping 'bounded' prefix from div scheme '{}'", key);
                tl.remove(0);
                tl.reset();
            }
        }
        catch (const std::bad_any_cast&)
        {}
    }
}

NeoN::Dictionary mapFvSchemes(const NeoN::Dictionary& schemesDict)
{
    NeoN::Dictionary modSchemesDict = schemesDict;

    updateDdtSchemes(modSchemesDict);
    stripBoundedDivSchemes(modSchemesDict);
    // snGrad scheme names (corrected, uncorrected, limited [corrected] <coeff>)
    // are accepted by NeoN's factories directly — no remapping required.
    // gradSchemes likewise need no remapping: OpenFOAM's "Gauss <interp>" and
    // "cellLimited Gauss <interp> <coeff>" token lists are consumed verbatim by
    // NeoN's GradOperatorFactory (the cellLimited factory wraps the base scheme
    // and reads the trailing limiter coefficient).

    return modSchemesDict;
}

} // namespace NeoFOAM
