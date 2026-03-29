// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2026 NeoFOAM authors


#include "NeoFOAM/compatibility/fvSchemes.hpp"

#include <map>

#include <NeoN/core/logging.hpp>
#include <NeoN/core/primitives/scalar.hpp>
#include <NeoN/core/primitives/label.hpp>


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

void updateSnGradSchemes(NeoN::Dictionary& schemeDict)
{
    if (!schemeDict.contains("snGradSchemes"))
    {
        return;
    }

    NeoN::Dictionary& snGradSchemes = schemeDict.subDict("snGradSchemes");

    // Map for TokenList-valued entries: rename first token (the scheme name)
    // "corrected" and "uncorrected" match NeoN names directly, only "limited" needs renaming
    static const std::map<std::string, std::string> tokenSchemeMap = {
        {"limited", "limitedCorrected"},
    };

    for (const auto& key : snGradSchemes.keys())
    {
        if (snGradSchemes.isType<NeoN::TokenList>(key))
        {
            auto& tl = snGradSchemes.get<NeoN::TokenList>(key);
            if (!tl.empty())
            {
                auto& firstName = tl.get<std::string>(0);
                auto it = tokenSchemeMap.find(firstName);
                if (it != tokenSchemeMap.end())
                {
                    NeoN::Logging::warn(
                        "Replacing snGrad scheme '{}' → '{}' for entry '{}'",
                        firstName,
                        it->second,
                        key
                    );
                    firstName = it->second;
                }
            }
        }
    }
}

NeoN::Dictionary mapFvSchemes(const NeoN::Dictionary& schemesDict)
{
    NeoN::Dictionary modSchemesDict = schemesDict;

    updateDdtSchemes(modSchemesDict);
    updateSnGradSchemes(modSchemesDict);

    return modSchemesDict;
}

} // namespace NeoFOAM
