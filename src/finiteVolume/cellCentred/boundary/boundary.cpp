// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/boundary.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"

namespace NeoFOAM
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::VolumeBoundaryFactory<scalar>;
template class fvcc::VolumeBoundaryFactory<Vector>;

template class fvcc::volumeBoundary::FixedValue<scalar>;
template class fvcc::volumeBoundary::FixedValue<Vector>;

template class fvcc::volumeBoundary::FixedGradient<scalar>;
template class fvcc::volumeBoundary::FixedGradient<Vector>;

template class fvcc::volumeBoundary::Calculated<scalar>;
template class fvcc::volumeBoundary::Calculated<Vector>;

template class fvcc::volumeBoundary::Empty<scalar>;
template class fvcc::volumeBoundary::Empty<Vector>;

template class fvcc::SurfaceBoundaryFactory<scalar>;
template class fvcc::SurfaceBoundaryFactory<Vector>;

template class fvcc::surfaceBoundary::FixedValue<scalar>;
template class fvcc::surfaceBoundary::FixedValue<Vector>;

template class fvcc::surfaceBoundary::FixedGradient<scalar>;
template class fvcc::surfaceBoundary::FixedGradient<Vector>;

template class fvcc::surfaceBoundary::Calculated<scalar>;
template class fvcc::surfaceBoundary::Calculated<Vector>;

template class fvcc::surfaceBoundary::Empty<scalar>;
template class fvcc::surfaceBoundary::Empty<Vector>;

}
