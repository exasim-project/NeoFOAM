// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/boundary/boundaryStrategy.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

// NOTE the int is currently just a dummy
template class fvcc::BoundaryFactory<NeoFOAM::scalar, int>;
template class fvcc::BoundaryFactory<NeoFOAM::Vector, int>;
