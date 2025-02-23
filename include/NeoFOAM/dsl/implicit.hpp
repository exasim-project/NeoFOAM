// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/dsl/temporalOperator.hpp"
#include "NeoFOAM/dsl/ddt.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;
namespace dsl = NeoFOAM::dsl;

namespace NeoFOAM::dsl::imp
{


TemporalOperator ddt(fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return fvcc::DdtOperator(dsl::Operator::Type::Implicit, phi);
}

SpatialOperator
Source(fvcc::VolumeField<NeoFOAM::scalar>& coeff, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator(fvcc::SourceTerm(dsl::Operator::Type::Implicit, coeff, phi));
}

SpatialOperator
div(fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator(fvcc::DivOperator(dsl::Operator::Type::Implicit, faceFlux, phi));
}

SpatialOperator
laplacian(fvcc::SurfaceField<NeoFOAM::scalar>& gamma, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator(fvcc::LaplacianOperator(dsl::Operator::Type::Implicit, gamma, phi));
}

} // namespace NeoFOAM
