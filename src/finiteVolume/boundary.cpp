// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/boundary.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"

namespace NeoFOAM
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::VolumeBoundaryFactory<scalar>;
template class fvcc::VolumeBoundaryFactory<Vector>;

template class fvcc::FixedValue<scalar>;
template class fvcc::FixedValue<Vector>;

template class fvcc::FixedGradient<scalar>;
template class fvcc::FixedGradient<Vector>;

template class fvcc::Calculated<scalar>;
template class fvcc::Calculated<Vector>;

template class fvcc::Empty<scalar>;
template class fvcc::Empty<Vector>;

}
