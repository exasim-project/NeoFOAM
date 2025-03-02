// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// instantiate the template class
template class FaceNormalGradient<scalar>;


// void Uncorrected::faceNormalGrad(
//     const VolumeField<scalar>& volField, SurfaceField<scalar>& surfaceField
// ) const
// {
//     computeFaceNormalGrad(volField, geometryScheme_, surfaceField);
// }

// const SurfaceField<scalar>& Uncorrected::deltaCoeffs() const
// {
//     return geometryScheme_->nonOrthDeltaCoeffs();
// }


// std::unique_ptr<FaceNormalGradientFactory<scalar>> Uncorrected::clone() const
// {
//     return std::make_unique<Uncorrected>(*this);
// }

} // namespace NeoFOAM
