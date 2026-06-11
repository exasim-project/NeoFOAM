// SPDX-FileCopyrightText: 2024 - 2026 NeoFOAM authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "NeoN/NeoN.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/**
 * @brief Compute the deviatoric stress field for the momentum equation:
 *        tau_ij = (nu + nut) * (dev2(gradU^T))_ij
 *               = (nu + nut) * (gradU_{ji} - (2/3)*tr(gradU)*delta_ij)
 *
 * Matches OpenFOAM's: nuEff * dev2(T(gradU))
 * Returns VolumeField<Tensor> (non-symmetric in general).
 * Used by divDevReff for the momentum viscous term.
 */
VolumeField<Tensor> computeDevStress(
    const VolumeField<scalar>& nu,
    const VolumeField<scalar>& nut,
    const VolumeField<Tensor>& gradU
);

/**
 * @brief Gauss divergence of the deviatoric stress field → momentum RHS.
 *
 * Interpolates tau to faces, accumulates -tau_f & S_f into rhs (normalised by cell volume).
 * Uses SurfaceInterpolation<Tensor>.
 */
void divDevReff(
    const SurfaceInterpolation<Tensor>& surfInterp,
    const VolumeField<Tensor>& tau,
    Vector<Vec3>& rhs,
    dsl::Coeff operatorScaling
);

} // namespace NeoN::finiteVolume::cellCentred
