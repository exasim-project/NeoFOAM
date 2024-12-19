// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/timeIntegration/timeIntegration.hpp"
#include "NeoFOAM/timeIntegration/forwardEuler.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::timeIntegration
{

template class ForwardEuler<fvcc::VolumeField<scalar>>;

} // namespace NeoFOAM::dsl
