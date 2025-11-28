// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <type_traits>

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/auxiliary/convert.hpp"
#include "NeoFOAM/auxiliary/typeConversion.hpp"
#include "NeoFOAM/auxiliary/fieldTraits.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
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
        std::cout << __FILE__ << __LINE__ << "readVolBoundaryConditions()\n";
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
             // test if things can be read as scalar first, if it doesn't work
             // read as int and convert to scalar
             if constexpr (std::is_same<type_primitive_t, NeoN::Vec3>::value)
             {
                 NeoN::Vec3 tmpFixedValue {};
                 auto tokens = tokenList.tokens();
                 NeoN::scalar* ret = std::any_cast<NeoN::scalar>(&tokens[1]);
                 if (ret)
                 {
                     tmpFixedValue[0] = tokenList.get<NeoN::scalar>(1);
                     tmpFixedValue[1] = tokenList.get<NeoN::scalar>(2);
                     tmpFixedValue[2] = tokenList.get<NeoN::scalar>(3);
                 }
                 else
                 {
                     tmpFixedValue[0] = NeoN::scalar(tokenList.get<int>(1));
                     tmpFixedValue[1] = NeoN::scalar(tokenList.get<int>(2));
                     tmpFixedValue[2] = NeoN::scalar(tokenList.get<int>(3));
                 }
                 dict.insert("fixedValue", tmpFixedValue);
             }
             else
             {
                 auto& token = tokenList.tokens()[1];
                 NeoN::scalar* ret = std::any_cast<NeoN::scalar>(&token);
                 fixedValue = ret ? NeoN::scalar(*ret) : NeoN::scalar(std::any_cast<int>(token));
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
        {"processor", [](auto& dict) { dict.insert("type", std::string("processor")); }},
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
        std::cout << __FILE__ << __LINE__ << "boundary " <<patchDict.get<Foam::word>("type") << " \n";
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
        std::cout << __FILE__ << __LINE__ << "constructFrom()\n";
    using type_container_t = typename TypeMap<FoamType>::container_type;
    using type_primitive_t = typename TypeMap<FoamType>::mapped_type;

    type_container_t out(exec, in.name(), nfMesh, readVolBoundaryConditions(nfMesh, in));

        std::cout << __FILE__ << __LINE__ << "fromFoamField()\n";
    out.internalVector() = fromFoamField(exec, in.primitiveField());
        std::cout << __FILE__ << __LINE__ << "correctBoundary()\n";
    out.correctBoundaryConditions();
        std::cout << __FILE__ << __LINE__ << " done correctBoundary()\n";

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
        std::cout << "map patch type" << patchDict.get<Foam::word>("type") << "\n";
        patchInserter[patchDict.get<Foam::word>("type")](neoPatchDict);
        bcs.push_back(fvcc::SurfaceBoundary<type_primitive_t>(uMesh, neoPatchDict, patchi));
        patchi++;
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
        ContainerType out(exec, in.name(), nfMesh, readVolBoundaryConditions(nfMesh, in));
        out.internalVector() = fromFoamField(exec, in.primitiveField());
        out.correctBoundaryConditions();
        return out;
    }
    else if constexpr (NeoFOAM::detail::isSurfaceField<ContainerType>)
    {
        using FoamComponentType = typename FoamFieldType::cmptType;

        ContainerType out(exec, in.name(), nfMesh, readSurfaceBoundaryConditions(nfMesh, in));

        const std::size_t nInt = nfMesh.nInternalFaces();
        const std::size_t nBnd = nfMesh.boundaryMesh().offset().back();
        const std::size_t nFaces = nInt + nBnd;

        NF_DINFO(
            "Internal: " + std::to_string(nInt) + ", Boundary: " + std::to_string(nBnd)
            + ", nFaces: " + std::to_string(nFaces)
        );

        Foam::Field<FoamComponentType> flat(nFaces);
        Foam::Field<FoamComponentType> bval(nBnd);

        // Internal faces first: [0, nInt)
        forAll(in, facei)
        {
            if (static_cast<std::size_t>(facei) < nInt)
            {
                flat[facei] = convert(in[facei]);
            }
        }

        // Boundary faces appended in patch order: [nInt, nFaces)
        Foam::label idx = static_cast<Foam::label>(nInt);
        Foam::label bi = 0;
        forAll(in.boundaryField(), patchi)
        {
            const auto& pin = in.boundaryField()[patchi];
            forAll(pin, facei)
            {
                flat[idx] = convert(pin[facei]);
                bval[bi] = convert(pin[facei]);
                ++idx;
                ++bi;
            }
        }

        // (Optional) asserts in debug:
        NF_ASSERT_EQUAL(static_cast<std::size_t>(idx), nFaces);
        NF_ASSERT_EQUAL(static_cast<std::size_t>(bi), nBnd);

        out.internalVector() = fromFoamField(exec, flat);
        out.boundaryData().value() = fromFoamField(exec, bval);
        out.correctBoundaryConditions();
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
        std::cout << __FILE__ << __LINE__ << "operator()\n";
        using type_container_t = typename TypeMap<FieldType>::container_type;
        type_container_t convertedField = constructFrom(exec, nfMesh, foamField);
        if (name != "")
        {
            convertedField.name = name;
        }
        const Foam::fvMesh& mesh = foamField.mesh();
        const Foam::Time& runTime = mesh.time();
        std::int64_t timeIndex = runTime.timeIndex();

        std::cout << __FILE__ << __LINE__ << "field()\n";
        NeoN::Field<typename type_container_t::VectorValueType> field(
            convertedField.exec(),
            convertedField.internalVector(),
            convertedField.boundaryData()
        );

        std::cout << __FILE__ << __LINE__ << "registeredField()\n";
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
