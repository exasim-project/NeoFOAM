// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2026 NeoFOAM authors


#include "NeoFOAM/compatibility/fvSchemes.hpp"

#include <map>
#include <memory>

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

NeoN::Dictionary mapFvSchemes(const NeoN::Dictionary& schemesDict)
{
    NeoN::Dictionary modSchemesDict = schemesDict;

    updateDdtSchemes(modSchemesDict);
    // div scheme names, including the "bounded <scheme>" convection wrapper, and snGrad
    // scheme names (corrected, uncorrected, limited [corrected] <coeff>) are accepted by
    // NeoN's factories directly — no remapping required.
    // gradSchemes likewise need no remapping: OpenFOAM's "Gauss <interp>" and
    // "cellLimited Gauss <interp> <coeff>" token lists are consumed verbatim by
    // NeoN's GradOperatorFactory (the cellLimited factory wraps the base scheme
    // and reads the trailing limiter coefficient).

    return modSchemesDict;
}

std::unique_ptr<fvcc::GradOperatorFactory<NeoN::Vec3>> makeGradOperator(
    const NeoN::Executor& exc,
    const NeoN::UnstructuredMesh& mesh,
    const NeoN::Dictionary& fvSchemes,
    const std::string& gradEntry
)
{
    NeoN::TokenList tokens;
    bool found = false;
    if (fvSchemes.contains("gradSchemes"))
    {
        const auto& gradSchemes = fvSchemes.subDict("gradSchemes");
        // A single-word scheme (e.g. "leastSquares") is stored as a std::string rather than a
        // TokenList, so it has to be wrapped before the factory sees it -- the same
        // normalisation expandSchemeDefaults applies. Reading it as a TokenList directly, or
        // skipping string entries, would silently discretise with the fallback scheme instead.
        auto readEntry = [&](const std::string& key)
        {
            if (!gradSchemes.contains(key)) return false;
            if (gradSchemes.isType<std::string>(key))
            {
                const auto& word = gradSchemes.get<std::string>(key);
                // "none" declares that no scheme applies; picking one anyway would change the
                // discretisation without saying so, so require an explicit entry instead.
                NF_ASSERT(
                    word != "none",
                    "gradSchemes entry '" << key << "' is 'none'; add an explicit " << gradEntry
                                          << " entry."
                );
                tokens = NeoN::TokenList({word});
            }
            else
            {
                tokens = gradSchemes.get<NeoN::TokenList>(key);
            }
            return true;
        };
        found = readEntry(gradEntry) || readEntry("default");
    }
    if (!found)
    {
        tokens = NeoN::TokenList({std::string("Gauss"), std::string("linear")});
    }
    // Rewind the read cursor: the dictionary's stored TokenList may have been
    // advanced by a previous create(), and create() reads from the cursor.
    tokens.reset();
    return fvcc::GradOperatorFactory<NeoN::Vec3>::create(exc, mesh, tokens);
}

} // namespace NeoFOAM
