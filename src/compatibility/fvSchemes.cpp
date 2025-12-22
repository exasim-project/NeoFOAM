// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
/* This file implements comparison operator to compare OpenFOAM and corresponding NeoFOAM fields
 * TODO the comparison operator only make sense for testing purposes
 * so this should be part of the tests
 */


#include "NeoFOAM/compatibility/fvSolution.hpp"

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

NeoN::Dictionary mapFvSchemes(const NeoN::Dictionary& schemesDict)
{
    NeoN::Dictionary modSchemesDict = schemesDict;

    updateDdtSchemes(modSchemesDict);

    return modSchemesDict;
}

} // namespace NeoFOAM
