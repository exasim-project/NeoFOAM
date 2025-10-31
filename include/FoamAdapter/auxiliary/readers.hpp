// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 FoamAdapter authors
#pragma once

#include <type_traits>

#include "NeoN/NeoN.hpp"

#include "FoamAdapter/auxiliary/convert.hpp"
#include "FoamAdapter/auxiliary/type_conversion.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace FoamAdapter
{

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
             dict.insert("fixedGradient", type_primitive_t {});
         }},
        {"fixedValue",
         [](auto& dict)
         {
             dict.insert("type", std::string("fixedValue"));
             NeoN::TokenList tokenList = dict.template get<NeoN::TokenList>("value");
             type_primitive_t fixedValue {};
             if (std::is_same<type_primitive_t, NeoN::Vec3>::value)
             {
                 NeoN::Vec3 tmpFixedValue {};
                 tmpFixedValue[0] = tokenList.get<int>(1);
                 tmpFixedValue[1] = tokenList.get<int>(2);
                 tmpFixedValue[2] = tokenList.get<int>(3);
                 dict.insert("fixedValue", tmpFixedValue);
             }
             else
             {
                 try
                 {
                     fixedValue = tokenList.get<type_primitive_t>(1);
                 }
                 catch (const std::bad_any_cast& e)
                 {
                     fixedValue = NeoN::one<type_primitive_t>() * (tokenList.get<int>(1));
                 }
                 dict.insert("fixedValue", fixedValue);
             }
         }},
        {"noSlip", // TODO specialize for vector
         [](auto& dict)
         {
             dict.insert("type", std::string("fixedValue"));
             dict.insert("fixedValue", type_primitive_t {});
         }},
        {"calculated", [](auto& dict) { dict.insert("type", std::string("calculated")); }},
        {"extrapolatedCalculated",
         [](auto& dict) { dict.insert("type", std::string("calculated")); }},
        {"empty", [](auto& dict) { dict.insert("type", std::string("empty")); }},
	{"symmetryPlane", [](auto& dict) { dict.insert("type", std::string("symmetry")); }},
        {"symmetry", [](auto& dict) { dict.insert("type", std::string("symmetry")); }}
    };

    int patchi = 0;
    std::vector<fvcc::VolumeBoundary<type_primitive_t>> bcs;
    for (const auto& bName : bDict.toc())
    {
        Foam::dictionary patchDict = bDict.subDict(bName);
        NeoN::Dictionary neoPatchDict = convert(patchDict);
        patchInserter[patchDict.get<Foam::word>("type")](neoPatchDict);
        bcs.emplace_back(nfMesh, neoPatchDict, patchi);
        patchi++;
    }
    return bcs;
}

template<typename FoamType>
auto constructFrom(
    const NeoN::Executor exec,
    const NeoN::UnstructuredMesh& nfMesh,
    const FoamType& in
)
{
    using type_container_t = typename TypeMap<FoamType>::container_type;
    using type_primitive_t = typename TypeMap<FoamType>::mapped_type;

    type_container_t out(exec, in.name(), nfMesh, readVolBoundaryConditions(nfMesh, in));

    out.internalVector() = fromFoamField(exec, in.primitiveField());
    out.correctBoundaryConditions();

    return out;
};

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
        {"calculated", [](auto& dict) { dict.insert("type", std::string("calculated")); }},
        {"empty", [](auto& dict) { dict.insert("type", std::string("empty")); }},
	{"symmetryPlane", [](auto& dict) { dict.insert("type", std::string("symmetry")); }},
        {"symmetry", [](auto& dict) { dict.insert("type", std::string("symmetry")); }}
    };

    for (const auto& bName : bDict.toc())
    {
        Foam::dictionary patchDict = bDict.subDict(bName);
        NeoN::Dictionary neoPatchDict;
        patchInserter[patchDict.get<Foam::word>("type")](neoPatchDict);
        bcs.push_back(fvcc::SurfaceBoundary<type_primitive_t>(uMesh, neoPatchDict, patchi));
        patchi++;
    }
    return bcs;
}

template<typename FoamType>
auto constructSurfaceField(
    const NeoN::Executor exec,
    const NeoN::UnstructuredMesh& nfMesh,
    const FoamType& in
)
{
    using type_container_t = typename TypeMap<FoamType>::container_type;
    using type_primitive_t = typename TypeMap<FoamType>::mapped_type;
    using foam_primitive_t = typename FoamType::cmptType;

    type_container_t
        out(exec, in.name(), nfMesh, std::move(readSurfaceBoundaryConditions(nfMesh, in)));

    Foam::Field<foam_primitive_t> flattenedField(out.internalVector().size());
    size_t nInternal = nfMesh.nInternalFaces();

    forAll(in, facei)
    {
        flattenedField[facei] = convert(in[facei]);
    }

    Foam::label idx = nInternal;
    Foam::Field<foam_primitive_t> bvalue(out.internalVector().size());
    forAll(in.boundaryField(), patchi)
    {
        const Foam::fvsPatchField<foam_primitive_t>& pin = in.boundaryField()[patchi];

        forAll(pin, facei)
        {
            flattenedField[idx] = pin[facei];
            bvalue[idx - nInternal] = pin[facei];
            idx++;
        }
    }
    assert(idx == flattenedField.size());

    out.internalVector() = fromFoamField(exec, flattenedField);
    out.boundaryData().value() = fromFoamField(exec, bvalue);
    out.correctBoundaryConditions();

    return out;
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

}; // namespace Foam
