// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/operator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/surfaceField.hpp"

// TODO we should get rid of this include since it includes details
// from a general implementation
#include "NeoFOAM/finiteVolume/cellCentred/operators/divOperator.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::dsl::exp
{

Operator
div(const fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return Operator(fvcc::DivOperator(dsl::Operator::Type::Explicit, faceFlux, phi));
}


} // namespace NeoFOAM
