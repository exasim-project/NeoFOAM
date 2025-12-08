// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
/* This file implements comparison operator to compare OpenFOAM and corresponding NeoFOAM fields
 * TODO the comparison operator only make sense for testing purposes
 * so this should be part of the tests
 */
#pragma once

#include "NeoN/NeoN.hpp"

#include "GeometricField.H"
#include "volMesh.H"
#include "Field.H"

namespace NeoFOAM
{
namespace fvcc = NeoN::finiteVolume::cellCentred;

template<typename NT, typename OT>
bool operator==(const NeoN::Vector<NT>& nf, const Foam::Field<OT>& of);

template<typename NT, typename OT>
bool operator==(
    fvcc::VolumeField<NT>& nf,
    const Foam::GeometricField<OT, Foam::fvPatchField, Foam::volMesh>& of
);

template<typename NT, typename OT>
bool operator==(
    const fvcc::SurfaceField<NT>& nf,
    const Foam::GeometricField<OT, Foam::fvsPatchField, Foam::surfaceMesh>& of
);

} // namespace Foam
