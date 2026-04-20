// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoN/NeoN.hpp"

#include "fvMesh.H"
#include "volFields.H"

#include "NeoFOAM/auxiliary/convert.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

namespace detail
{

/*@brief copy from neon src vector on device to dest OF field*/
template<class SrcValueType, class DestField>
void copyImpl(const NeoN::Vector<SrcValueType>& src, DestField& dest)
{
    NF_ASSERT_EQUAL(dest.size(), src.size());
    auto srcHost = src.copyToHost();
    auto srcView = srcHost.view();
    for (int i = 0; i < dest.size(); i++)
    {
        dest[i] = convert(srcView[i]);
    }
}

/*@brief copy from OF field to dest neon vector*/
template<class SrcField, class DestValueType>
void copyImplToNF(const SrcField& src, NeoN::Vector<DestValueType>& dest)
{
    NF_ASSERT_EQUAL(dest.size(), src.size());
    auto exec = dest.exec();
    auto tmpVec = NeoN::Vector<DestValueType>(
        exec,
        reinterpret_cast<const DestValueType*>(src.cdata()),
        src.size()
    );
    dest=tmpVec;
}

}

template<class SrcField>
void syncToFoam(const SrcField& src, Foam::volVectorField& dest)
{
    detail::copyImpl(src.internalVector(), dest.ref());

    auto hostBCValue = src.boundaryData().value().copyToHost();

    forAll(dest.boundaryField(), patchi)
    {
        auto& foamFieldPatch = dest.boundaryFieldRef()[patchi];
        auto [start, end] = src.boundaryData().range(patchi);

        // forAll(foamFieldPatch, bfacei)
        // {
        //     foamFieldPatch[bfacei] = hostBCValue.view()[start + bfacei];
        // }
    }
}

template<class SrcField, class DstField>
void syncFromFoam(const SrcField& src, DstField& dest)
{
    detail::copyImplToNF(src, dest.internalVector());

    // auto hostBCValue = src.boundaryData().value().copyToHost();

    // forAll(dest.boundaryField(), patchi)
    // {
    //     auto& foamFieldPatch = dest.boundaryFieldRef()[patchi];
    //     auto [start, end] = src.boundaryData().range(patchi);

    //     // forAll(foamFieldPatch, bfacei)
    //     // {
    //     //     foamFieldPatch[bfacei] = hostBCValue.view()[start + bfacei];
    //     // }
    // }
}

/*@brief writes a NeoN field back to disk using OF field file format*/
void write(const NeoN::scalarVector& sf, const Foam::fvMesh& mesh, const std::string fieldName);

/*@brief writes a NeoN field back to disk using OF field file format*/
void write(
    const NeoN::Vector<NeoN::Vec3>& sf,
    const Foam::fvMesh& mesh,
    const std::string fieldName
);

/*@brief writes a NeoN field back to disk using OF field file format*/
void write(const fvcc::VolumeField<NeoN::scalar>& volField, const Foam::fvMesh& mesh);

/*@brief writes a NeoN field back to disk using OF field file format*/
void write(const fvcc::VolumeField<NeoN::Vec3>& volField, const Foam::fvMesh& mesh);

} // namespace Foam
