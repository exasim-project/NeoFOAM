// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/primitives/scalar.hpp"
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


TemporalOperator<scalar> ddt(fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return fvcc::DdtOperator(dsl::Operator::Type::Implicit, phi);
}

SpatialOperator<scalar>
Source(fvcc::VolumeField<NeoFOAM::scalar>& coeff, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator<scalar>(fvcc::SourceTerm(dsl::Operator::Type::Implicit, coeff, phi));
}

SpatialOperator<scalar>
div(fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator<scalar>(fvcc::DivOperator(dsl::Operator::Type::Implicit, faceFlux, phi));
}

SpatialOperator<scalar>
laplacian(fvcc::SurfaceField<NeoFOAM::scalar>& gamma, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return SpatialOperator<scalar>(
        fvcc::LaplacianOperator(dsl::Operator::Type::Implicit, gamma, phi)
    );
}

} // namespace NeoFOAM
