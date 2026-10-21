// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

void updateDdtSchemes(NeoN::Dictionary& solverDict);

NeoN::Dictionary mapFvSchemes(const NeoN::Dictionary& schemesDict);

/**
 * @brief Build a runtime-selected gradient operator for a field from gradSchemes.
 *
 * Looks up @p gradEntry (e.g. "grad(U)") in the gradSchemes sub-dict, falling back
 * to the "default" entry, and finally to "Gauss linear". This lets the hardcoded
 * grad call sites (velocity correction, turbulence grad(U)/grad(nuTilda)) honour
 * the configured scheme — e.g. cellLimited — instead of always using Gauss-Green.
 *
 * The returned operator handles both the scalar gradient (grad(scalar) -> Vec3)
 * and the tensor gradient (gradTensor: grad(Vec3) -> Tensor).
 *
 * @param exc        Executor.
 * @param mesh       NeoN unstructured mesh.
 * @param fvSchemes  Mapped fvSchemes dictionary (see mapFvSchemes).
 * @param gradEntry  The gradSchemes key, e.g. "grad(U)" / "grad(p)" / "grad(nuTilda)".
 */
std::unique_ptr<fvcc::GradOperatorFactory<NeoN::Vec3>> makeGradOperator(
    const NeoN::Executor& exc,
    const NeoN::UnstructuredMesh& mesh,
    const NeoN::Dictionary& fvSchemes,
    const std::string& gradEntry
);

/**
 * @brief Fill missing specific scheme keys from the "default" entry in each scheme sub-dict.
 *
 * NeoN operators look up explicit keys (e.g. "laplacian(nuEff,U)") and do not fall back to
 * "default" themselves.  This function performs that expansion at the NeoFOAM level, using
 * the field/flux/gamma names embedded in the expression operators.
 *
 * @param schemesDict  The mapped fvSchemes dictionary (output of mapFvSchemes).
 * @param expr         The DSL expression whose operators supply field/flux/gamma names.
 * @param fieldName    The name of the primary solved field (psi), used for ddt/grad lookups
 *                     where the operator config is not available.
 * @return A copy of schemesDict with all missing specific keys pre-populated from "default".
 */
template<typename ValueType>
NeoN::Dictionary expandSchemeDefaults(
    const NeoN::Dictionary& schemesDict,
    const NeoN::dsl::Expression<ValueType>& expr,
    const std::string& fieldName
)
{
    NeoN::Dictionary resolved = schemesDict;

    auto fillDefault = [&](const std::string& subName, const std::string& key)
    {
        if (!resolved.contains(subName)) return;
        auto& sub = resolved.subDict(subName);
        if (!sub.contains(key) && sub.contains("default"))
        {
            if (sub.template isType<std::string>("default"))
            {
                const auto& defaultStr = sub.get<std::string>("default");
                if (defaultStr == "none")
                    return; // "none" means no fallback — require explicit entry
                NeoN::TokenList tl;
                tl.insert(defaultStr);
                sub.insert(key, tl);
            }
            else
                sub.insert(key, sub.get<NeoN::TokenList>("default"));
        }
    };

    // ddt: DdtOperator::read() expects std::string, not TokenList — insert directly
    if (resolved.contains("ddtSchemes"))
    {
        auto& ddtSub = resolved.subDict("ddtSchemes");
        const std::string ddtKey = "ddt(" + fieldName + ")";
        if (!ddtSub.contains(ddtKey) && ddtSub.contains("default")
            && ddtSub.template isType<std::string>("default"))
        {
            const auto& def = ddtSub.get<std::string>("default");
            if (def != "none") ddtSub.insert(ddtKey, def);
        }
    }
    fillDefault("gradSchemes", "grad(" + fieldName + ")");

    // div and laplacian need the flux/gamma names from each operator's config
    for (const auto& op : expr.spatialOperators())
    {
        if (op.getName() == "ViscousStressOperator")
        {
            fillDefault("divSchemes", "div((nuEff*dev2(T(grad(U)))))");
            continue;
        }

        if (op.getName() == "GradOperator")
        {
            auto config = op.getConfig();
            if (config.contains("field"))
            {
                const auto& scalarField =
                    config
                        .template get<NeoN::detail::RefHolder<fvcc::VolumeField<NeoN::scalar>>>(
                            "field"
                        )
                        .c;
                fillDefault("gradSchemes", "grad(" + scalarField.name + ")");
            }
            continue;
        }

        auto config = op.getConfig();
        if (!config.contains("field")) continue;

        const auto& field =
            config.template get<NeoN::detail::RefHolder<fvcc::VolumeField<ValueType>>>("field").c;

        if (op.getName() == "DivOperator" && config.contains("flux"))
        {
            const auto& flux =
                config
                    .template get<NeoN::detail::RefHolder<fvcc::SurfaceField<NeoN::scalar>>>("flux")
                    .c;
            fillDefault("divSchemes", "div(" + flux.name + "," + field.name + ")");
        }
        else if (op.getName() == "LaplacianOperator" && config.contains("gamma"))
        {
            const auto& gamma =
                config
                    .template get<NeoN::detail::RefHolder<fvcc::SurfaceField<NeoN::scalar>>>("gamma"
                    )
                    .c;
            fillDefault("laplacianSchemes", "laplacian(" + gamma.name + "," + field.name + ")");
        }
    }

    return resolved;
}

} // namespace NeoFOAM
