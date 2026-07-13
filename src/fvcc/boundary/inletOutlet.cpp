// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// SPDX-License-Identifier: MIT

#include "NeoFOAM/fvcc/boundary/volume/inletOutlet.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

template class fvcc::volumeBoundary::InletOutlet<NeoN::scalar>;
template class fvcc::volumeBoundary::InletOutlet<NeoN::Vec3>;
