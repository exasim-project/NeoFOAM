// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "volFields.H"
#include "surfaceFields.H"

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/auxiliary/convert.hpp"

namespace NeoFOAM
{

namespace fvcc = NeoN::finiteVolume::cellCentred;
template<typename From>
struct TypeMap
{
};

// Specializations of TypeMap for specific type mappings.
template<>
struct TypeMap<Foam::GeometricField<Foam::scalar, Foam::fvPatchField, Foam::volMesh>>
{
    using container_type = fvcc::VolumeField<NeoN::scalar>;
    using mapped_type = NeoN::scalar;
};

template<>
struct TypeMap<Foam::GeometricField<Foam::vector, Foam::fvPatchField, Foam::volMesh>>
{
    using container_type = fvcc::VolumeField<NeoN::Vec3>;
    using mapped_type = NeoN::Vec3;
};

template<>
struct TypeMap<Foam::GeometricField<Foam::tensor, Foam::fvPatchField, Foam::volMesh>>
{
    using container_type = fvcc::VolumeField<NeoN::Tensor>;
    using mapped_type = NeoN::Tensor;
};

template<>
struct TypeMap<Foam::GeometricField<Foam::scalar, Foam::fvsPatchField, Foam::surfaceMesh>>
{
    using container_type = fvcc::SurfaceField<NeoN::scalar>;
    using mapped_type = NeoN::scalar;
};

template<>
struct TypeMap<Foam::GeometricField<Foam::vector, Foam::fvsPatchField, Foam::surfaceMesh>>
{
    using container_type = fvcc::SurfaceField<NeoN::Vec3>;
    using mapped_type = NeoN::Vec3;
};

// Specializations of TypeMap for specific type mappings.
template<>
struct TypeMap<Foam::Field<Foam::scalar>>
{
    using container_type = NeoN::Vector<NeoN::scalar>;
    using mapped_type = NeoN::scalar;
};

// Specializations of TypeMap for specific type mappings.
template<>
struct TypeMap<Foam::Field<Foam::vector>>
{
    using container_type = NeoN::Vector<NeoN::Vec3>;
    using mapped_type = NeoN::Vec3;
};

// Foam::tensor and NeoN::Tensor are both nine contiguous row-major scalars, so the
// reinterpret_cast in fromFoamField is layout-compatible (as for scalar/vector).
template<>
struct TypeMap<Foam::Field<Foam::tensor>>
{
    using container_type = NeoN::Vector<NeoN::Tensor>;
    using mapped_type = NeoN::Tensor;
};

// Specializations of TypeMap for specific type mappings.
template<>
struct TypeMap<Foam::List<Foam::scalar>>
{
    using container_type = NeoN::Vector<NeoN::scalar>;
    using mapped_type = NeoN::scalar;
};

// Specializations of TypeMap for specific type mappings.
template<>
struct TypeMap<Foam::List<Foam::label>>
{
    using container_type = NeoN::Vector<NeoN::label>;
    using mapped_type = NeoN::label;
};

}; // namespace Foam
