// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryBase.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/surfaceBoundaryBase.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

template class fvcc::VolumeBoundaryModel<NeoFOAM::scalar>;
template class fvcc::VolumeBoundaryModel<NeoFOAM::Vector>;
