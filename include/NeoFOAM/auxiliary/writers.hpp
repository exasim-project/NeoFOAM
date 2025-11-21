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

/*@brief copy from neon src field on device to dest OF field*/
template<class SrcField, class DestField>
void copyImpl(const SrcField& src, DestField& dest)
{
    NF_ASSERT_EQUAL(dest.size(), src.size());
    auto srcHost = src.copyToHost();
    auto srcView = srcHost.view();
    for (int i = 0; i < dest.size(); i++)
    {
        dest[i] = convert(srcView[i]);
    }
}
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
void write(
    const fvcc::VolumeField<NeoN::scalar>& volField
    const Foam::fvMesh& mesh,
);

/*@brief writes a NeoN field back to disk using OF field file format*/
void write(
    const fvcc::VolumeField<NeoN::Vec3>& volField,
    const Foam::fvMesh& mesh
);

} // namespace Foam
