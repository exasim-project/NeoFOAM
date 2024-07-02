// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

template class fvcc::VolumeBoundaryFactory<NeoFOAM::scalar>;
template class fvcc::VolumeBoundaryFactory<NeoFOAM::Vector>;
