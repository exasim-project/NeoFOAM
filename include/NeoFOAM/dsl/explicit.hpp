// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/surfaceField.hpp"

// TODO we should get rid of this include since it includes details
// from a general implementation
#include "NeoFOAM/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/sourceTerm.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::dsl::exp
{


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
Source(fvcc::VolumeField<NeoFOAM::scalar>& coeff, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator<scalar>(fvcc::SourceTerm(dsl::Operator::Type::Explicit, coeff, phi));
}


} // namespace NeoFOAM
