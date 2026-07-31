// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <type_traits>

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/auxiliary/convert.hpp"
#include "NeoFOAM/auxiliary/typeConversion.hpp"
#include "NeoFOAM/auxiliary/fieldTraits.hpp"
#include "messageStream.H"
#include "processorFvPatch.H"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

namespace detail
{

/**
 * @brief Promote a single TokenList entry to a NeoN::scalar regardless of
 * whether OpenFOAM's lexer classified it as a SCALAR (double) or a LABEL (int).
 *
 * OpenFOAM tokenizes numeric primitives per token: a literal with a decimal
 * point (e.g. `0.1`) becomes a SCALAR while a bare integer (e.g. `0`) becomes a
 * LABEL. The components of a single vector value can therefore have different
 * stored types — `uniform (0.1 0 0)` yields [scalar, label, label]. A strict
 * `TokenList::get<NeoN::scalar>` throws `bad_any_cast` on the label components,
 * so each component must be probed and promoted independently.
 */
inline NeoN::scalar tokenAsScalar(NeoN::TokenList& tokenList, std::size_t idx)
{
    auto& tokens = tokenList.tokens();
    if (const NeoN::scalar* asScalar = std::any_cast<NeoN::scalar>(&tokens[idx]))
    {
        return *asScalar;
    }
    // Not stored as a scalar — it was parsed as an integer label. Promote it.
    return NeoN::scalar(tokenList.get<Foam::label>(idx));
}

/**
 * @brief Mean unit normal of an OpenFOAM patch and how far its faces deviate from it.
 *
 * NeoN's fixedValue carries a single value for the whole patch, so a boundary condition
 * defined per face along the surface normal has to collapse to one normal. `deviation` is
 * the largest |n_f - meanNormal| over the patch faces: 0 on a planar patch (where the
 * collapse is exact), O(1) on a strongly curved one.
 */
struct PatchNormal
{
    NeoN::Vec3 meanNormal;
    NeoN::scalar deviation;
};

inline PatchNormal meanUnitNormal(const Foam::fvPatch& patch)
{
    const Foam::vectorField faceNormals(patch.nf());
    if (faceNormals.empty())
    {
        return {NeoN::Vec3(0.0, 0.0, 0.0), 0.0};
    }

    Foam::vector mean(Foam::Zero);
    for (const Foam::vector& faceNormal : faceNormals)
    {
        mean += faceNormal;
    }
    // Re-normalise: averaging unit normals shortens the sum on a curved patch, and on a
    // closed one it cancels to (almost) zero — VSMALL keeps that finite, and the deviation
    // it produces then trips the caller's non-planar warning.
    mean /= Foam::mag(mean) + Foam::VSMALL;

    NeoN::scalar deviation = 0.0;
    for (const Foam::vector& faceNormal : faceNormals)
    {
        deviation = std::max(deviation, NeoN::scalar(Foam::mag(faceNormal - mean)));
    }
    return {convert(mean), deviation};
}

/**
 * @brief Pin a total-pressure patch at the value OpenFOAM evaluated when the field was read.
 *
 * Shared by `totalPressure` and `uniformTotalPressure`: NeoN models neither the dynamic-head
 * correction nor a time-dependent p0, so the patch becomes a fixedValue at the start-time
 * total pressure. Non-scalar (or valueless) falls back to zeroGradient.
 */
template<typename ValueType>
void insertFrozenTotalPressure(NeoN::Dictionary& dict)
{
    if constexpr (std::is_same<ValueType, NeoN::scalar>::value)
    {
        if (dict.contains("value"))
        {
            NeoN::TokenList tokenList = dict.get<NeoN::TokenList>("value");
            if (tokenList.size() > 1)
            {
                dict.insert("type", std::string("fixedValue"));
                dict.insert("fixedValue", tokenAsScalar(tokenList, 1));
                return;
            }
        }
    }
    dict.insert("type", std::string("fixedGradient"));
    dict.insert("fixedGradient", NeoN::zero<ValueType>());
}

} // namespace detail

template<typename FoamType>
auto fromFoamField(const NeoN::Executor& exec, const FoamType& field)
{
    using type_container_t = typename TypeMap<FoamType>::container_type;
    using mapped_t = typename TypeMap<FoamType>::mapped_type;
    type_container_t nfField(
        exec,
        reinterpret_cast<const mapped_t*>(field.cdata()),
        static_cast<size_t>(field.size())
    );
    return nfField;
};

template<typename FoamType>
auto readVolBoundaryConditions(const NeoN::UnstructuredMesh& nfMesh, const FoamType& ofVolField)
{
    using type_container_t = typename TypeMap<FoamType>::container_type;
    using type_primitive_t = typename TypeMap<FoamType>::mapped_type;

    // get boundary as dictionary
    Foam::OStringStream os;
    ofVolField.boundaryField().writeEntries(os);
    Foam::IStringStream is(os.str());
    Foam::dictionary bDict(is);

    // The inserters receive only the patch dictionary, but the translations below need the
    // patch itself — its geometry (surfaceNormalFixedValue) and its name (the approximation
    // notices). applyVolInserter points these at the patch it is about to translate. The
    // notices stream the name as a `const char*`: OpenFOAM's Ostream writes a std::string
    // as a quoted string token.
    std::string activePatchName;
    const Foam::fvPatch* activeFoamPatch = nullptr;

    std::map<std::string, std::function<void(NeoN::Dictionary&)>> patchInserter {
        {"fixedGradient",
         [](auto& dict)
         {
             dict.insert("type", std::string("fixedGradient"));
             NeoN::TokenList tokenList = dict.template get<NeoN::TokenList>("value");
             type_primitive_t fixedGradient = tokenList.get<type_primitive_t>(1);
             dict.insert("fixedGradient", fixedGradient);
         }},
        {"zeroGradient",
         [&](auto& dict)
         {
             dict.insert("type", std::string("fixedGradient"));
             dict.insert("fixedGradient", NeoN::zero<type_primitive_t>());
         }},
        {"fixedValue",
         [](auto& dict)
         {
             NeoN::TokenList tokenList = dict.template get<NeoN::TokenList>("value");
             auto fixedValue = NeoN::zero<type_primitive_t>();

             // NOTE FIXME in parallel cases we end up with token.size ==1
             // leading to nonuniform as first token. This  means probably that
             // parsing the foam dictionary aborts early and omits 0();
             if (tokenList.size() > 1)
             {
                 dict.insert("type", std::string("fixedValue"));
                 // OpenFOAM classifies each numeric token independently: a literal
                 // with a decimal point is a SCALAR, a bare integer is a LABEL. A
                 // single vector value can therefore mix the two (e.g. `(0.1 0 0)`
                 // -> [scalar, label, label]). Read every component via
                 // detail::tokenAsScalar so each is promoted to scalar on its own,
                 // instead of choosing one branch for the whole value based solely
                 // on the first component (which caused bad_any_cast on index 2/3).
                 if constexpr (std::is_same<type_primitive_t, NeoN::Vec3>::value)
                 {
                     NeoN::Vec3 tmpFixedValue {};
                     tmpFixedValue[0] = detail::tokenAsScalar(tokenList, 1);
                     tmpFixedValue[1] = detail::tokenAsScalar(tokenList, 2);
                     tmpFixedValue[2] = detail::tokenAsScalar(tokenList, 3);
                     dict.insert("fixedValue", tmpFixedValue);
                     return;
                 }
                 else
                 {
                     fixedValue = detail::tokenAsScalar(tokenList, 1);
                 }
                 dict.insert("fixedValue", fixedValue);
             }
             else
             {
                 // FIXME is this an empty boundary?
                 dict.insert("type", std::string("empty"));
                 // left blank
             }
         }},
        {"uniformFixedValue",
         [](auto& dict)
         {
             // uniformFixedValue stores its value as a Function1. We only support
             // the (by far most common) `constant <value>` form, which converts to
             // a TokenList of [word "constant", value...]; the value therefore
             // starts at index 1, exactly like the `value uniform ...` case above.
             dict.insert("type", std::string("fixedValue"));
             NeoN::TokenList tokenList = dict.template get<NeoN::TokenList>("uniformValue");
             if constexpr (std::is_same<type_primitive_t, NeoN::Vec3>::value)
             {
                 NeoN::Vec3 tmpFixedValue {};
                 tmpFixedValue[0] = detail::tokenAsScalar(tokenList, 1);
                 tmpFixedValue[1] = detail::tokenAsScalar(tokenList, 2);
                 tmpFixedValue[2] = detail::tokenAsScalar(tokenList, 3);
                 dict.insert("fixedValue", tmpFixedValue);
             }
             else
             {
                 dict.insert("fixedValue", detail::tokenAsScalar(tokenList, 1));
             }
         }},
        {"noSlip", // TODO specialize for vector
         [](auto& dict)
         {
             dict.insert("type", std::string("fixedValue"));
             dict.insert("fixedValue", type_primitive_t {});
         }},
        {"calculated", [](auto& dict) { dict.insert("type", std::string("calculated")); }},
        {"processor", [](auto& dict) { dict.insert("type", std::string("processor")); }},
        {"extrapolatedCalculated",
         [](auto& dict) { dict.insert("type", std::string("calculated")); }},
        {"empty", [](auto& dict) { dict.insert("type", std::string("empty")); }},
        {"symmetryPlane", [](auto& dict) { dict.insert("type", std::string("symmetry")); }},
        {"symmetry", [](auto& dict) { dict.insert("type", std::string("symmetry")); }},
        {"nutUSpaldingWallFunction",
         [](auto& dict) { dict.insert("type", std::string("nutUSpaldingWallFunction")); }},
        {"kqRWallFunction", [](auto& dict) { dict.insert("type", std::string("kqRWallFunction")); }
        },
        {"omegaWallFunction",
         [](auto& dict) { dict.insert("type", std::string("omegaWallFunction")); }},
        {"epsilonWallFunction",
         [](auto& dict) { dict.insert("type", std::string("epsilonWallFunction")); }},
        {"nutkWallFunction",
         [](auto& dict) { dict.insert("type", std::string("nutkWallFunction")); }},
        {"inletOutlet",
         [](auto& dict)
         {
             dict.insert("type", std::string("inletOutlet"));
             NeoN::TokenList tokenList = dict.template get<NeoN::TokenList>("inletValue");
             if constexpr (std::is_same<type_primitive_t, NeoN::Vec3>::value)
             {
                 NeoN::Vec3 inletValue {};
                 if (tokenList.size() >= 4)
                 {
                     inletValue[0] = detail::tokenAsScalar(tokenList, 1);
                     inletValue[1] = detail::tokenAsScalar(tokenList, 2);
                     inletValue[2] = detail::tokenAsScalar(tokenList, 3);
                 }
                 dict.insert("inletValue", inletValue);
             }
             else
             {
                 dict.insert(
                     "inletValue",
                     tokenList.size() >= 2 ? detail::tokenAsScalar(tokenList, 1)
                                           : type_primitive_t {}
                 );
             }
         }},
        // Two-phase VoF outlet/pressure conditions from the damBreak case. These
        // are outflow-dominant and are approximated here as zeroGradient so the
        // fields can be read; a faithful treatment lands with the VoF solve.
        {"pressureInletOutletVelocity",
         [&](auto& dict)
         {
             dict.insert("type", std::string("fixedGradient"));
             dict.insert("fixedGradient", NeoN::zero<type_primitive_t>());
         }},
        {"fixedFluxPressure",
         [&](auto& dict)
         {
             // Faithful wall fixedFluxPressure: the per-face refGrad is set externally by
             // NeoFOAM::constrainPressure so the projection cancels the buoyancy/capillary
             // wall face flux. The FixedFluxPressure BC ignores the dict (refGrad is
             // zero-initialised -> zeroGradient until the first constrainPressure).
             dict.insert("type", std::string("fixedFluxPressure"));
         }},
        {"totalPressure",
         [&](auto& dict)
         {
             // p_rgh atmosphere: pin the pressure datum with a fixedValue at the stored
             // patch value so the p_rgh solve is well-posed. The dynamic-head correction
             // is negligible for the gravity-driven damBreak first version and is deferred
             // with surface tension. Non-scalar (or valueless) falls back to zeroGradient.
             detail::insertFrozenTotalPressure<type_primitive_t>(dict);
         }},
        {"uniformTotalPressure",
         [&](auto& dict)
         {
             WarningInFunction
                 << "uniformTotalPressure on patch '" << activePatchName.c_str()
                 << "' is approximated as a fixedValue at the total pressure OpenFOAM"
                    " evaluated for the start time.\n    NeoN applies neither the p0(t)"
                    " Function1 nor the dynamic-head correction p0 - 0.5|U|^2, so the patch"
                    " pressure stays constant for the whole run."
                 << Foam::endl;
             detail::insertFrozenTotalPressure<type_primitive_t>(dict);
         }},
        {"slip", [](auto& dict) { dict.insert("type", std::string("slip")); }},
        {"movingWallVelocity",
         [&](auto& dict)
         {
             WarningInFunction
                 << "movingWallVelocity on patch '" << activePatchName.c_str()
                 << "' is approximated as a stationary no-slip wall, fixedValue (0 0 0).\n"
                    "    NeoN has no mesh motion, so the wall velocity U_wall = U_mesh = 0;"
                    " the result differs from OpenFOAM wherever the mesh actually moves."
                 << Foam::endl;
             dict.insert("type", std::string("fixedValue"));
             dict.insert("fixedValue", NeoN::zero<type_primitive_t>());
         }},
        {"surfaceNormalFixedValue",
         [&](auto& dict)
         {
             // OpenFOAM evaluates refValue*n_f per face; NeoN's fixedValue holds one value
             // for the whole patch, so refValue is projected onto the patch's mean unit
             // normal here — exact on a planar patch (every intake in the sweep), an
             // approximation on a curved one, which is why the deviation is checked.
             if constexpr (std::is_same<type_primitive_t, NeoN::Vec3>::value)
             {
                 if (activeFoamPatch == nullptr)
                 {
                     throw std::runtime_error(
                         "surfaceNormalFixedValue on patch '" + activePatchName
                         + "': patch geometry not found in the mesh."
                     );
                 }
                 NeoN::TokenList tokenList = dict.template get<NeoN::TokenList>("refValue");
                 const std::string* form = tokenList.size() > 1
                                             ? std::any_cast<std::string>(&tokenList.tokens()[0])
                                             : nullptr;
                 if (form == nullptr || *form != "uniform")
                 {
                     throw std::runtime_error(
                         "surfaceNormalFixedValue on patch '" + activePatchName
                         + "': only a uniform refValue is supported."
                     );
                 }
                 const NeoN::scalar refValue = detail::tokenAsScalar(tokenList, 1);
                 const auto [meanNormal, deviation] = detail::meanUnitNormal(*activeFoamPatch);

                 if (dict.contains("ramp"))
                 {
                     WarningInFunction
                         << "surfaceNormalFixedValue on patch '" << activePatchName.c_str()
                         << "' carries a `ramp` Function1, which NeoN does not model: the full"
                            " refValue is applied from the first time step."
                         << Foam::endl;
                 }
                 if (deviation > 1e-6)
                 {
                     WarningInFunction
                         << "surfaceNormalFixedValue on patch '" << activePatchName.c_str()
                         << "' is not planar (face normals deviate by up to " << deviation
                         << " from the patch mean): the single fixedValue refValue*n uses the"
                            " mean normal for every face."
                         << Foam::endl;
                 }
                 dict.insert("type", std::string("fixedValue"));
                 dict.insert("fixedValue", refValue * meanNormal);
             }
             else
             {
                 throw std::runtime_error(
                     "surfaceNormalFixedValue on patch '" + activePatchName
                     + "' is only defined for vector fields."
                 );
             }
         }}
    };

    auto applyVolInserter =
        [&](const std::string& patchName, const std::string& bcType, NeoN::Dictionary& dict)
    {
        auto it = patchInserter.find(bcType);
        if (it == patchInserter.end())
        {
            std::string supported;
            for (const auto& [key, _] : patchInserter)
                supported += "\n  " + key;
            throw std::runtime_error(
                "Unsupported boundary condition type '" + bcType + "' on patch '" + patchName
                + "'.\nSupported types:" + supported
            );
        }
        activePatchName = patchName;
        const Foam::label foamPatchID = ofVolField.mesh().boundaryMesh().findPatchID(patchName);
        activeFoamPatch = foamPatchID >= 0 ? &ofVolField.mesh().boundary()[foamPatchID] : nullptr;
        it->second(dict);
    };

    int patchi = 0;
    std::vector<fvcc::VolumeBoundary<type_primitive_t>> bcs;
    // do non processor first
    for (const auto bName : bDict.toc())
    {
        Foam::dictionary patchDict = bDict.subDict(bName);
        std::string bcType = patchDict.get<Foam::word>("type");
        if (bcType != "processor")
        {
            NeoN::Dictionary neoPatchDict = convert(patchDict);
            applyVolInserter(bName, bcType, neoPatchDict);
            bcs.emplace_back(nfMesh, neoPatchDict, patchi);
            patchi++;
        }
    }
    for (const auto bName : bDict.toc())
    {
        Foam::dictionary patchDict = bDict.subDict(bName);
        std::string bcType = patchDict.get<Foam::word>("type");
        if (bcType == "processor")
        {
            NeoN::Dictionary neoPatchDict = convert(patchDict);
            applyVolInserter(bName, bcType, neoPatchDict);
            bcs.emplace_back(nfMesh, neoPatchDict, patchi);
            patchi++;
        }
    }
    return bcs;
}

template<typename FoamType>
auto readSurfaceBoundaryConditions(
    const NeoN::UnstructuredMesh& uMesh,
    const FoamType& surfaceField
)
{
    using type_container_t = typename TypeMap<FoamType>::container_type;
    using type_primitive_t = typename TypeMap<FoamType>::mapped_type;

    std::vector<fvcc::SurfaceBoundary<type_primitive_t>> bcs;

    // get boundary as dictionary
    Foam::OStringStream os;
    surfaceField.boundaryField().writeEntries(os);
    Foam::IStringStream is(os.str());
    Foam::dictionary bDict(is);
    int patchi = 0;

    // TODO this approach fails for procBoundary0to1
    std::map<std::string, std::function<void(NeoN::Dictionary&)>> patchInserter {
        {"fixedGradient", [](auto& dict) { dict.insert("type", std::string("fixedGradient")); }},
        {"zeroGradient",
         [&](auto& dict)
         {
             dict.insert("type", std::string("fixedGradient"));
             dict.insert("fixedGradient", type_primitive_t {});
         }},
        {"fixedValue",
         [](auto& dict)
         {
             dict.insert("type", std::string("fixedValue"));
             dict.insert("fixedValue", type_primitive_t {});
         }},
        {"noSlip", // TODO specialize for vector
         [](auto& dict)
         {
             dict.insert("type", std::string("fixedValue"));
             dict.insert("fixedValue", type_primitive_t {});
         }},
        {"calculated", [](auto& dict) { dict.insert("type", std::string("calculated")); }},
        {"processor", [](auto& dict) { dict.insert("type", std::string("processor")); }},
        {"empty", [](auto& dict) { dict.insert("type", std::string("empty")); }},
        {"symmetryPlane", [](auto& dict) { dict.insert("type", std::string("symmetry")); }},
        {"symmetry", [](auto& dict) { dict.insert("type", std::string("symmetry")); }}
    };

    auto applySurfaceInserter =
        [&](const std::string& patchName, const std::string& bcType, NeoN::Dictionary& dict)
    {
        auto it = patchInserter.find(bcType);
        if (it == patchInserter.end())
        {
            std::string supported;
            for (const auto& [key, _] : patchInserter)
                supported += "\n  " + key;
            throw std::runtime_error(
                "Unsupported boundary condition type '" + bcType + "' on patch '" + patchName
                + "'.\nSupported types:" + supported
            );
        }
        it->second(dict);
    };

    for (const auto& bName : bDict.toc())
    {
        Foam::dictionary patchDict = bDict.subDict(bName);
        std::string bcType = patchDict.get<Foam::word>("type");
        if (bcType != "processor")
        {
            NeoN::Dictionary neoPatchDict;
            applySurfaceInserter(bName, bcType, neoPatchDict);
            bcs.push_back(fvcc::SurfaceBoundary<type_primitive_t>(uMesh, neoPatchDict, patchi));
            patchi++;
        }
    }
    for (const auto& bName : bDict.toc())
    {
        Foam::dictionary patchDict = bDict.subDict(bName);
        std::string bcType = patchDict.get<Foam::word>("type");
        if (bcType == "processor")
        {
            NeoN::Dictionary neoPatchDict;
            applySurfaceInserter(bName, bcType, neoPatchDict);
            bcs.push_back(fvcc::SurfaceBoundary<type_primitive_t>(uMesh, neoPatchDict, patchi));
            patchi++;
        }
    }
    return bcs;
}

template<class FoamFieldType>
auto constructFrom(
    const NeoN::Executor exec,
    const NeoN::UnstructuredMesh& nfMesh,
    const FoamFieldType& in
)
{
    using ContainerType = typename TypeMap<FoamFieldType>::container_type;
    using MappedType = typename TypeMap<FoamFieldType>::mapped_type;

    if constexpr (NeoFOAM::detail::isVolumeField<ContainerType>)
    {
        using FoamValueType = typename FoamFieldType::value_type;
        ContainerType out(exec, in.name(), nfMesh, readVolBoundaryConditions(nfMesh, in));
        out.internalVector() = fromFoamField(exec, in.primitiveField());
        std::size_t nBnd = 0;
        forAll(in.boundaryField(), patchi)
        {
            nBnd += in.boundaryField()[patchi].size();
        }

        Foam::Field<FoamValueType> bval(nBnd);

        Foam::label bi = 0;
        forAll(in.boundaryField(), patchi)
        {
            const auto& pin = in.boundaryField()[patchi];
            forAll(pin, i)
            {
                // IMPORTANT:
                // keep OpenFOAM type here (scalar or vector)
                bval[bi++] = pin[i];
            }
        }

        NF_ASSERT_EQUAL(static_cast<std::size_t>(bi), nBnd);
        out.boundaryData().value() = fromFoamField(exec, bval);
        out.correctBoundaryConditions();
        return out;
    }
    else if constexpr (NeoFOAM::detail::isSurfaceField<ContainerType>)
    {
        // Element type of the GeometricField (T in GeometricField<T,...>) — NOT cmptType,
        // which decomposes vectors into scalars and would break the surface-vector path.
        using FoamComponentType = typename FoamFieldType::value_type;

        ContainerType out(exec, in.name(), nfMesh, readSurfaceBoundaryConditions(nfMesh, in));

        const std::size_t nInt = nfMesh.nInternalFaces();
        const std::size_t nBnd = nfMesh.boundaryMesh().offset().back();

        NF_DINFO("Internal: " + std::to_string(nInt) + ", Boundary: " + std::to_string(nBnd));

        Foam::Field<FoamComponentType> internalData(nInt);
        Foam::Field<FoamComponentType> bval(nBnd);

        // Internal faces only: [0, nInt)
        forAll(in, facei)
        {
            if (static_cast<std::size_t>(facei) < nInt)
            {
                internalData[facei] = in[facei];
            }
        }

        // Boundary faces in patch order: [0, nBnd)
        Foam::label bi = 0;
        // Pass 1 — non-processor patches.
        forAll(in.boundaryField(), patchi)
        {
            const auto& pin = in.boundaryField()[patchi];
            if (pin.patch().type() == "processor")
            {
                continue;
            }
            forAll(pin, facei)
            {
                bval[bi] = pin[facei];
                ++bi;
            }
        }
        // Pass 2 — processor patches (proc tail of the boundary range).
        forAll(in.boundaryField(), patchi)
        {
            const auto& pin = in.boundaryField()[patchi];
            if (pin.patch().type() != "processor")
            {
                continue;
            }
            forAll(pin, facei)
            {
                bval[bi] = pin[facei];
                ++bi;
            }
        }

        NF_ASSERT_EQUAL(static_cast<std::size_t>(bi), nBnd);

        out.internalVector() = fromFoamField(exec, internalData);
        out.boundaryData().value() = fromFoamField(exec, bval);
        return out;
    }
    else
    {
        NF_ASSERT(
            (!std::is_same_v<ContainerType, ContainerType>),
            "TypeMap<FoamFieldType>::container_type must be VolumeField<ValueType> or "
            "SurfaceField<ValueType>."
        );
    }
}

/**
 * @brief Creates a VectorDocument from an existing Foam Field.
 *

 * @return The created VectorDocument.
 */
template<typename FieldType>
class CreateFromFoamField
{
public:

    const NeoN::Executor exec;
    const NeoN::UnstructuredMesh& nfMesh;
    const FieldType& foamField;
    std::string name = "";
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = -1;

    fvcc::VectorDocument operator()(NeoN::Database& db)
    {
        using type_container_t = typename TypeMap<FieldType>::container_type;

        type_container_t convertedField = constructFrom(exec, nfMesh, foamField);

        if (name != "")
        {
            convertedField.name = name;
        }
        const Foam::fvMesh& mesh = foamField.mesh();
        const Foam::Time& runTime = mesh.time();
        std::int64_t timeIndex = runTime.timeIndex();

        NeoN::Field<typename type_container_t::VectorValueType> field(
            convertedField.exec(),
            convertedField.internalVector(),
            convertedField.boundaryData()
        );

        type_container_t registeredField(
            convertedField.exec(),
            convertedField.name,
            convertedField.mesh(),
            field,
            convertedField.boundaryConditions(),
            db,
            "",
            ""
        );

        return NeoN::Document(
            {{"name", convertedField.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", registeredField}},
            fvcc::validateVectorDoc
        );
    }
};

/**
 * @brief Constructs a set of NN fields from OF fields
 * @tparam Types Types of the classes
 * @return Tuple containing the fields
 */
template<typename... Types>
auto constFromMany(const NeoN::Executor& exec, const NeoN::UnstructuredMesh& uMesh, Types&... args)
{
    return std::tuple(constructFrom(exec, uMesh, args)...);
}

/**@brief construct an NF field from a given OF field and register*/
template<typename FoamFieldType>
auto& constructAndRegister(
    fvcc::VectorCollection& fieldCollection,
    RunTime& rt,
    const FoamFieldType& of,
    bool storeOldTime = true
)
{
    using ContainerType = typename TypeMap<FoamFieldType>::container_type;
    ContainerType& ret = fieldCollection.template registerVector<ContainerType>(
        NeoFOAM::CreateFromFoamField<FoamFieldType> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = of,
            .name = of.name()
        }
    );
    if (storeOldTime)
    {
        fvcc::rotateOldTimes(ret);
    }
    ret.correctBoundaryConditions();
    return ret;
}


}; // namespace Foam
