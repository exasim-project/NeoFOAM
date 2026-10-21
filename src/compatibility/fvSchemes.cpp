// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2026 NeoFOAM authors


#include "NeoFOAM/compatibility/fvSchemes.hpp"

#include <any>
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

// The reconstruction gradient of linearUpwind arrives as a *key* into gradSchemes, e.g.
// "div(phi,U) bounded Gauss linearUpwind limited" alongside "limited cellLimited Gauss
// linear 1". NeoN holds no gradSchemes table, so it cannot resolve that key and falls back
// to the unlimited Gauss-Green gradient -- an unclipped, systematically more aggressive
// convection term. fvSchemes lives here, so expand the key to its definition first.
//
// A spec that already names its gradient inline ("linearUpwind cellLimited Gauss linear 1")
// is left alone, as is one whose key has no gradSchemes entry.
static void expandLinearUpwindGradSchemes(NeoN::Dictionary& schemeDict)
{
    if (!schemeDict.contains("divSchemes") || !schemeDict.contains("gradSchemes")) return;

    const NeoN::Dictionary& gradSchemes = schemeDict.subDict("gradSchemes");
    NeoN::Dictionary& divSchemes = schemeDict.subDict("divSchemes");

    for (const auto& key : divSchemes.keys())
    {
        if (!divSchemes.isType<NeoN::TokenList>(key)) continue;
        NeoN::TokenList& tl = divSchemes.get<NeoN::TokenList>(key);

        auto& toks = tl.tokens();
        for (std::size_t i = 0; i + 1 < toks.size(); ++i)
        {
            if (toks[i].type() != typeid(std::string)) continue;
            const auto& word = std::any_cast<const std::string&>(toks[i]);
            if (word != "linearUpwind" && word != "linearUpwindV") continue;

            if (toks[i + 1].type() != typeid(std::string)) break;
            const auto gradKey = std::any_cast<const std::string&>(toks[i + 1]);

            // Already an inline gradient spec rather than a key.
            if (gradKey == "Gauss" || gradKey == "cellLimited" || gradKey == "leastSquares") break;

            // A key with no entry of its own resolves through "default", exactly as
            // OpenFOAM's mesh.gradScheme() does -- without this a spec like
            // "linearUpwind grad(U)" whose gradSchemes only defines "default" would be left
            // unresolved and NeoN would fall back to its unlimited gradient.
            std::string gradEntry = gradKey;
            if (!gradSchemes.contains(gradEntry)) gradEntry = "default";
            if (!gradSchemes.contains(gradEntry)) break;

            std::vector<std::any> expanded;
            if (gradSchemes.isType<NeoN::TokenList>(gradEntry))
            {
                // get() on a const Dictionary yields a const TokenList, whose tokens() is
                // non-const; copy through a mutable one.
                NeoN::TokenList g = gradSchemes.get<NeoN::TokenList>(gradEntry);
                expanded = g.tokens();
            }
            else if (gradSchemes.isType<std::string>(gradEntry))
            {
                expanded.emplace_back(gradSchemes.get<std::string>(gradEntry));
            }
            else
            {
                break;
            }

            NeoN::Logging::warn(
                "Expanding linearUpwind gradient '{}' in div scheme '{}' to its gradSchemes "
                "definition '{}'",
                gradKey,
                key,
                gradEntry
            );
            toks.erase(toks.begin() + static_cast<std::ptrdiff_t>(i) + 1);
            toks.insert(
                toks.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                expanded.begin(),
                expanded.end()
            );
            tl.reset();
            break;
        }
    }
}

// limitedLinear builds its TVD ratio from the gradient of the transported field, which
// OpenFOAM resolves through gradSchemes as grad(<field>) (LimitedScheme.C calls the one-argument
// fvc::grad, and limitFuncs::magSqr<scalar> passes a scalar field through unchanged, so the
// lookup key carries the field's own name). NeoN's limitedLinear has no gradSchemes table and
// would always use the unlimited Gauss-Green gradient; that reports a larger upwind-cell slope,
// hence a larger TVD ratio, a limiter nearer 1 and a blend nearer pure central differencing.
//
// So resolve grad(<field>) here -- <field> being the second argument of "div(phi,<field>)" -- and
// append the marker NeoN's spec accepts when it is a cellLimited scheme. A div key that is not of
// that form, or a field with no cellLimited gradient entry, is left alone.
static void appendLimitedLinearGradSchemes(NeoN::Dictionary& schemeDict)
{
    if (!schemeDict.contains("divSchemes") || !schemeDict.contains("gradSchemes")) return;

    const NeoN::Dictionary& gradSchemes = schemeDict.subDict("gradSchemes");
    NeoN::Dictionary& divSchemes = schemeDict.subDict("divSchemes");

    for (const auto& key : divSchemes.keys())
    {
        if (!divSchemes.isType<NeoN::TokenList>(key)) continue;

        // "div(phi,k)" -> "k"; anything else (div((nuEff*dev2(T(grad(U)))))) is not a transported
        // field and has no gradSchemes entry to find.
        const auto comma = key.find(',');
        if (key.rfind("div(", 0) != 0 || comma == std::string::npos || key.back() != ')') continue;
        const std::string field = key.substr(comma + 1, key.size() - comma - 2);
        if (field.empty() || field.find('(') != std::string::npos) continue;

        NeoN::TokenList& tl = divSchemes.get<NeoN::TokenList>(key);
        auto& toks = tl.tokens();

        // The coefficient follows the scheme name, so the marker goes at the very end; only act
        // when limitedLinear is the last *word* in the spec.
        bool hasLimitedLinear = false;
        for (const auto& tok : toks)
        {
            if (tok.type() != typeid(std::string)) continue;
            hasLimitedLinear = std::any_cast<const std::string&>(tok) == "limitedLinear";
        }
        if (!hasLimitedLinear) continue;

        // A field without its own entry falls back to "default", exactly as OpenFOAM's
        // mesh.gradScheme() does.
        std::string gradKey = "grad(" + field + ")";
        if (!gradSchemes.contains(gradKey)) gradKey = "default";
        if (!gradSchemes.contains(gradKey)) continue;

        // The entry may be a key into gradSchemes itself ("grad(k) $limited") -- OpenFOAM expands
        // those macros while reading, so by the time it reaches here it is the definition.
        std::string firstWord;
        if (gradSchemes.isType<NeoN::TokenList>(gradKey))
        {
            NeoN::TokenList g = gradSchemes.get<NeoN::TokenList>(gradKey);
            if (!g.tokens().empty() && g.tokens().front().type() == typeid(std::string))
            {
                firstWord = std::any_cast<const std::string&>(g.tokens().front());
            }
        }
        else if (gradSchemes.isType<std::string>(gradKey))
        {
            firstWord = gradSchemes.get<std::string>(gradKey);
        }
        if (firstWord != "cellLimited") continue;

        NeoN::Logging::warn(
            "Marking limitedLinear in div scheme '{}' as cellLimited, following gradSchemes "
            "entry '{}'",
            key,
            gradKey
        );
        toks.emplace_back(std::string("cellLimited"));
        tl.reset();
    }
}

NeoN::Dictionary mapFvSchemes(const NeoN::Dictionary& schemesDict)
{
    NeoN::Dictionary modSchemesDict = schemesDict;

    updateDdtSchemes(modSchemesDict);
    // The "bounded <scheme>" convection prefix is handed to NeoN untouched: BoundedDiv
    // implements it, subtracting the Sp(div(phi), psi) term that compensates a non-zero
    // continuity error in a steady run. Stripping the prefix silently dropped that term.
    expandLinearUpwindGradSchemes(modSchemesDict);
    appendLimitedLinearGradSchemes(modSchemesDict);
    // snGrad scheme names (corrected, uncorrected, limited [corrected] <coeff>)
    // are accepted by NeoN's factories directly — no remapping required.
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
