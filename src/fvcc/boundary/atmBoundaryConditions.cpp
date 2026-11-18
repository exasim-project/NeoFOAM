// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Instantiates the atmospheric boundary layer BCs in libNeoFOAM and forces their CRTP
// self-registration, so readers.hpp finds them by name however a solver is built. The
// non-template wall functions are otherwise only registered in translation units that happen to
// include their headers, and the inlet template not at all.

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/fvcc/boundary/volume/atmBoundaryLayerInlet.hpp"
#include "NeoFOAM/fvcc/boundary/volume/atmEpsilonWallFunction.hpp"
#include "NeoFOAM/fvcc/boundary/volume/atmNutkWallFunction.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace vb = NeoN::finiteVolume::cellCentred::volumeBoundary;

template class vb::AtmBoundaryLayerInlet<NeoN::scalar>;
template class vb::AtmBoundaryLayerInlet<NeoN::Vec3>;

namespace
{
using ScalarVolBCFactory = fvcc::VolumeBoundaryFactory<NeoN::scalar>;
using VectorVolBCFactory = fvcc::VolumeBoundaryFactory<NeoN::Vec3>;

[[maybe_unused]] const bool* const registerAtmBCs[] = {
    &ScalarVolBCFactory::Register<vb::AtmBoundaryLayerInlet<NeoN::scalar>>::REGISTERED,
    &VectorVolBCFactory::Register<vb::AtmBoundaryLayerInlet<NeoN::Vec3>>::REGISTERED,
    &ScalarVolBCFactory::Register<vb::AtmNutkWallFunction>::REGISTERED,
    &ScalarVolBCFactory::Register<vb::AtmEpsilonWallFunction>::REGISTERED,
};
} // namespace
