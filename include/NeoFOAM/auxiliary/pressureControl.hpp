// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include <optional>

#include "NeoN/NeoN.hpp"

namespace NeoFOAM
{

/// The pressure-limit entries of a solver's PIMPLE dictionary; an unset entry is absent.
struct PressureLimitSettings
{
    std::optional<NeoN::scalar> pMin;
    std::optional<NeoN::scalar> pMax;
    std::optional<NeoN::scalar> pMinFactor;
    std::optional<NeoN::scalar> pMaxFactor;
    std::optional<NeoN::scalar> rhoMin;
    std::optional<NeoN::scalar> rhoMax;
};

/**
 * @brief Pressure limits of the compressible solvers, as OpenFOAM's Foam::pressureControl.
 *
 * Resolves the limits like pressureControl's constructor with pRefRequired = false (what
 * rhoPimpleFoam uses): an absolute pMin/pMax pair wins; otherwise pMax (pMin) is the explicit
 * value, pMaxFactor (pMinFactor) times the largest (smallest) boundary value of p over the
 * patches that fix its value, or, for backward compatibility, the pressure belonging to
 * rhoMax (rhoMin) at the reference density. The factor and rho forms need at least one
 * value-fixing p patch and throw otherwise, as OpenFOAM does.
 *
 * limit() then clips p where it leaves [pMin, pMax], like pressureControl::limit(). The caller
 * re-evaluates p's boundary conditions afterwards when limit() returns true.
 */
class PressureControl
{
public:

    PressureControl(
        const NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>& p,
        const NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>& rho,
        const PressureLimitSettings& settings
    );

    /// True if pMin and/or pMax is set. limit() returns this value.
    bool active() const { return limitMinP_ || limitMaxP_; }

    bool limitsMin() const { return limitMinP_; }
    bool limitsMax() const { return limitMaxP_; }
    NeoN::scalar pMin() const { return pMin_; }
    NeoN::scalar pMax() const { return pMax_; }

    /**
     * @brief Clip the internal values of p to [pMin, pMax].
     *
     * The extremes are taken over the internal AND boundary values, as OpenFOAM's max(p) does,
     * and p is only touched when one of them is outside the limits. Boundary values are left
     * alone: the caller evaluates p's boundary conditions afterwards, which is what recomputes
     * them (a value-fixing patch keeps its value, as OpenFOAM's assignment to it is a no-op).
     *
     * @return true if limits are configured (even if nothing was clipped), like OpenFOAM.
     */
    bool limit(NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>& p) const;

private:

    NeoN::scalar pMin_ = 0;
    NeoN::scalar pMax_ = 0;
    bool limitMinP_ = false;
    bool limitMaxP_ = false;
};

} // namespace NeoFOAM
