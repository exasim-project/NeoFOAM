// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once


#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/dsl/expression.hpp"


namespace NeoN::finiteVolume::cellCentred
{

std::tuple<VolumeField<scalar>, VolumeField<Vector>>
discreteMomentumFields(const Expression<Vector>& expr);

void updateFaceVelocity(
    SurfaceField<scalar> phi,
    const SurfaceField<scalar> predictedPhi,
    const Expression<scalar>& expr
);

void updateVelocity(
    VolumeField<Vector>& u,
    const VolumeField<Vector>& hbyA,
    VolumeField<scalar>& rAU,
    VolumeField<scalar>& p
);

}
