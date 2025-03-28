// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/dsl/explicit.hpp"

namespace NeoFOAM::dsl::exp
{
namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

SpatialOperator<scalar>
div(const fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator<scalar>(fvcc::DivOperator(dsl::Operator::Type::Explicit, faceFlux, phi));
}

SpatialOperator<scalar> div(const fvcc::SurfaceField<NeoFOAM::scalar>& flux)
{
    return SpatialOperator<scalar>(fvcc::SurfaceIntegrate<scalar>(flux));
}

SpatialOperator<scalar>
source(fvcc::VolumeField<NeoFOAM::scalar>& coeff, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator<scalar>(fvcc::SourceTerm(dsl::Operator::Type::Explicit, coeff, phi));
}

} // namespace NeoFOAM
