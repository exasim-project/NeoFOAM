// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
#pragma once

#include "NeoN/NeoN.hpp"

namespace NeoFOAM {
namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace detail {

// Surface vs Volume detection on the **container type** produced by TypeMap.
template <class ContainerType> struct IsSurfaceField : std::false_type {};
template <class ValueType> struct IsSurfaceField<fvcc::SurfaceField<ValueType>> : std::true_type {};

template <class ContainerType>
inline constexpr bool isSurfaceField = IsSurfaceField<ContainerType>::value;

template <class ContainerType> struct IsVolumeField : std::false_type {};
template <class ValueType> struct IsVolumeField<fvcc::VolumeField<ValueType>> : std::true_type {};

template <class ContainerType>
inline constexpr bool isVolumeField = IsVolumeField<ContainerType>::value;

}  // namespace detail
}  // namespace NeoFOAM
