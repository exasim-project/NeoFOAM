// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include <optional>

namespace NeoFOAM
{

/**
 * @brief Per-mesh scratch reused across bound() calls.
 *
 * The per-cell sum of face areas is a function of the MESH only, so it is built once and reused
 * for every field bounded on that mesh. Hold one of these next to the fields being bounded; a
 * single cache serves every field on the same mesh.
 */
struct BoundCache
{
    std::optional<NeoN::Vector<NeoN::scalar>> sumFaceArea;
};

/**
 * @brief Bound a scalar field from below, following OpenFOAM's Foam::bound().
 *
 * A plain clamp to the lower bound is not equivalent and is actively harmful for the turbulence
 * scalars: a cell whose solve undershoots to zero or below is pinned at a floor many orders below
 * the physical scale of the field, and a subsequent quotient such as nu_t = Cmu k^2/eps then
 * explodes. OpenFOAM instead REFILLS non-positive cells with the local neighbourhood value —
 * fvc::average(max(vsf, lowerBound)), the face-area-weighted average of the linearly interpolated
 * bounded field — and applies the floor only as an outer max. Cells that are positive but below
 * the bound keep their value and are raised to the floor, exactly as in OpenFOAM.
 *
 * The min test and the reported extrema are global, so a distributed run bounds and reports the
 * same way a serial one does.
 *
 * @param vsf         [in,out] field to bound
 * @param lowerBound  the lower bound
 * @param cache       [in,out] mesh-derived scratch, reused across calls
 * @return true if the field was out of bounds and has been modified
 */
// Defined in src/auxiliary/bound.cpp, not inline here: its NEON_LAMBDA kernels would otherwise be
// instantiated in every including translation unit, and nvcc's extended-lambda wrappers are not
// ODR-safe across them (the linker keeps one copy of the inline function whose lambda wrapper may
// belong to another TU and is never initialized -> null function pointer on the device launch).
bool bound(
    NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>& vsf,
    const NeoN::scalar lowerBound,
    BoundCache& cache
);

/**
 * @brief bound() without a caller-held cache; rebuilds the mesh scratch on every call.
 *
 * Prefer the caching overload on any per-iteration path.
 */
inline bool bound(
    NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>& vsf,
    const NeoN::scalar lowerBound
)
{
    BoundCache scratch;
    return bound(vsf, lowerBound, scratch);
}

} // namespace NeoFOAM
