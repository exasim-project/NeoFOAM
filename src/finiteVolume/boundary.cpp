// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/boundary/boundaryBase.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryBase.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/surfaceBoundaryBase.hpp"

#include "NeoFOAM/finiteVolume/cellCentred.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

template class fvcc::VolumeBoundaryFactory<NeoFOAM::scalar>;
template class fvcc::VolumeBoundaryFactory<NeoFOAM::Vector>;

template class fvcc::FixedValue<NeoFOAM::scalar>;
template class fvcc::FixedValue<NeoFOAM::Vector>;

template class fvcc::FixedGradient<NeoFOAM::scalar>;
template class fvcc::FixedGradient<NeoFOAM::Vector>;

template class fvcc::Calculated<NeoFOAM::scalar>;
template class fvcc::Calculated<NeoFOAM::Vector>;

template class fvcc::Empty<NeoFOAM::scalar>;
template class fvcc::Empty<NeoFOAM::Vector>;
