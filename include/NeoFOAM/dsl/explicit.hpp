// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/surfaceField.hpp"

// TODO we should get rid of this include since it includes details
// from a general implementation
#include "NeoFOAM/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/sourceTerm.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::dsl::exp
{

SpatialOperator
div(const fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator(fvcc::DivOperator(dsl::SpatialOperator::Type::Explicit, faceFlux, phi));
}

SpatialOperator
Source(fvcc::VolumeField<NeoFOAM::scalar>& coeff, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator(fvcc::SourceTerm(dsl::SpatialOperator::Type::Explicit, coeff, phi));
}


} // namespace NeoFOAM
