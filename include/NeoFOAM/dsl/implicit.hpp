// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/operator.hpp"
#include "NeoFOAM/dsl/ddt.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;
namespace dsl = NeoFOAM::dsl;

namespace NeoFOAM::dsl::imp
{


Operator ddt(fvcc::VolumeField<NeoFOAM::scalar>& phi) { return dsl::temporal::ddt(phi); }

Operator Source(fvcc::VolumeField<NeoFOAM::scalar>& coeff, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return Operator(fvcc::SourceTerm(dsl::Operator::Type::Implicit, coeff, phi));
}

Operator div(fvcc::SurfaceField<NeoFOAM::scalar>& faceFlux, fvcc::VolumeField<NeoFOAM::scalar>& phi)
{
    return Operator(fvcc::DivOperator(dsl::Operator::Type::Implicit, faceFlux, phi));
}

} // namespace NeoFOAM
