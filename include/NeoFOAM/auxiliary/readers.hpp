// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <type_traits>

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/auxiliary/convert.hpp"
#include "NeoFOAM/auxiliary/typeConversion.hpp"
#include "NeoFOAM/auxiliary/fieldTraits.hpp"

#include "processorFvPatch.H"

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
                         tmpFixedValue[0] = NeoN::scalar(tokenList.get<Foam::label>(1));
                         tmpFixedValue[1] = NeoN::scalar(tokenList.get<Foam::label>(2));
                         tmpFixedValue[2] = NeoN::scalar(tokenList.get<Foam::label>(3));
                     }
                     dict.insert("fixedValue", tmpFixedValue);
                     return;
                 }
                 else
                 {
                     auto tokens = tokenList.tokens();
                     fixedValue = 0.0;
                     NeoN::scalar* ret = std::any_cast<NeoN::scalar>(&tokens[1]);
                     fixedValue =
                         ret ? NeoN::scalar(*ret) : NeoN::scalar(tokenList.get<Foam::label>(1));
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
    NeoN::mpi::Environment mpiEnv;
    // do non processor first
    for (const auto bName : bDict.toc())
    {
        Foam::dictionary patchDict = bDict.subDict(bName);
        if (patchDict.get<Foam::word>("type") != "processor")
        {
            NeoN::Dictionary neoPatchDict = convert(patchDict);
            patchInserter[patchDict.get<Foam::word>("type")](neoPatchDict);
            bcs.emplace_back(nfMesh, neoPatchDict, patchi);
            patchi++;
        }
    }
    for (const auto bName : bDict.toc())
    {
        Foam::dictionary patchDict = bDict.subDict(bName);
        if (patchDict.get<Foam::word>("type") == "processor")
        {
            NeoN::Dictionary neoPatchDict = convert(patchDict);
            patchInserter[patchDict.get<Foam::word>("type")](neoPatchDict);
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

    for (const auto& bName : bDict.toc())
    {
        Foam::dictionary patchDict = bDict.subDict(bName);
        if (patchDict.get<Foam::word>("type") != "processor")
        {
            NeoN::Dictionary neoPatchDict;
            patchInserter[patchDict.get<Foam::word>("type")](neoPatchDict);
            bcs.push_back(fvcc::SurfaceBoundary<type_primitive_t>(uMesh, neoPatchDict, patchi));
            patchi++;
        }
    }
    for (const auto& bName : bDict.toc())
    {
        Foam::dictionary patchDict = bDict.subDict(bName);
        if (patchDict.get<Foam::word>("type") == "processor")
        {
            NeoN::Dictionary neoPatchDict;
            patchInserter[patchDict.get<Foam::word>("type")](neoPatchDict);
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
        NeoN::mpi::Environment mpiEnv;
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

        NF_DINFO("Internal: " + std::to_string(nInt) + ", Boundary: " + std::to_string(nBnd));

        Foam::Field<FoamComponentType> internalData(nInt);
        Foam::Field<FoamComponentType> bval(nBnd);

        // Internal faces only: [0, nInt)
        forAll(in, facei)
        {
            if (static_cast<std::size_t>(facei) < nInt)
            {
                internalData[facei] = convert(in[facei]);
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
                bval[bi] = convert(pin[facei]);
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
                bval[bi] = convert(pin[facei]);
                ++bi;
            }
        }

        NF_ASSERT_EQUAL(static_cast<std::size_t>(bi), nBnd);

        out.internalVector() = fromFoamField(exec, internalData);
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
