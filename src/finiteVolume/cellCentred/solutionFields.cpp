// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#include "NeoFOAM/finiteVolume/cellCentred/solutionFields.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


SolutionFields::SolutionFields
(
    const VolumeField<scalar>& field
)
    : field(field), fieldName_(field.name)
{}

   

} // namespace NeoFOAM
