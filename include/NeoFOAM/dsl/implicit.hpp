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

} // namespace NeoFOAM
