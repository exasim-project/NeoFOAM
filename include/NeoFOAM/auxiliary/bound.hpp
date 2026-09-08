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
inline bool bound(
    NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>& vsf,
    const NeoN::scalar lowerBound,
    BoundCache& cache
)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    using NeoN::localIdx;
    using NeoN::scalar;

    const auto exec = vsf.exec();
    const auto& mesh = vsf.mesh();
    const auto nCells = vsf.internalVector().size();

    auto vsfV = vsf.internalVector().view();

    scalar minVsf = std::numeric_limits<scalar>::max();
    Kokkos::Min<scalar> minReducer(minVsf);
    NeoN::parallelReduce(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx i, scalar& m) { m = Kokkos::min(m, vsfV[i]); },
        minReducer
    );

    scalar maxVsf = std::numeric_limits<scalar>::lowest();
    Kokkos::Max<scalar> maxReducer(maxVsf);
    NeoN::parallelReduce(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx i, scalar& m) { m = Kokkos::max(m, vsfV[i]); },
        maxReducer
    );

    scalar sumVsf = 0.0;
    NeoN::parallelReduce(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx i, scalar& s) { s += vsfV[i]; },
        sumVsf
    );
    scalar cellCount = static_cast<scalar>(nCells);

#ifdef NF_WITH_MPI_SUPPORT
    if (mesh.boundaryMesh().isDistributed())
    {
        NeoN::mpi::Environment env;
        MPI_Allreduce(MPI_IN_PLACE, &minVsf, 1, NeoN::mpi::getType<scalar>(), MPI_MIN, env.comm());
        MPI_Allreduce(MPI_IN_PLACE, &maxVsf, 1, NeoN::mpi::getType<scalar>(), MPI_MAX, env.comm());
        scalar sums[2] = {sumVsf, cellCount};
        MPI_Allreduce(MPI_IN_PLACE, sums, 2, NeoN::mpi::getType<scalar>(), MPI_SUM, env.comm());
        sumVsf = sums[0];
        cellCount = sums[1];
    }
#endif

    if (minVsf >= lowerBound)
    {
        return false;
    }

    NeoN::Logging::info(
        "bounding {}, min: {} max: {} average: {}",
        vsf.name,
        minVsf,
        maxVsf,
        sumVsf / cellCount
    );

    // fvc::average(max(vsf, lowerBound)): face-area-weighted cell average of the linearly
    // interpolated bounded field. Accumulated per face over owner and neighbour, matching
    // surfaceSum(magSf*ssf)/surfaceSum(magSf) over internal AND boundary faces.
    NeoN::Vector<scalar> num(exec, nCells, scalar(0));
    auto numV = num.view();

    const bool buildSumFaceArea =
        !cache.sumFaceArea.has_value() || cache.sumFaceArea->size() != nCells;
    if (buildSumFaceArea)
    {
        cache.sumFaceArea.emplace(exec, nCells, scalar(0));
    }
    auto denV = cache.sumFaceArea->view();

    const auto geo = fvcc::GeometryScheme::readOrCreate(mesh);
    const auto [wS, areaS, ownerS, neighS] = NeoN::views(
        geo->weights().internalVector(),
        mesh.faceAreas(),
        mesh.faceOwners(),
        mesh.faceNeighbors()
    );

    const bool buildSum = buildSumFaceArea;
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx facei) {
            const auto own = ownerS[facei];
            const auto nei = neighS[facei];
            const scalar bOwn = Kokkos::max(vsfV[own], lowerBound);
            const scalar bNei = Kokkos::max(vsfV[nei], lowerBound);
            const scalar faceValue = wS[facei] * bOwn + (scalar(1) - wS[facei]) * bNei;
            const scalar aSf = areaS[facei];
            Kokkos::atomic_add(&numV[own], aSf * faceValue);
            Kokkos::atomic_add(&numV[nei], aSf * faceValue);
            if (buildSum)
            {
                Kokkos::atomic_add(&denV[own], aSf);
                Kokkos::atomic_add(&denV[nei], aSf);
            }
        },
        "bound::averageInternal"
    );

    const auto& bMesh = mesh.boundaryMesh();
    const auto [bOwners, bAreas] = NeoN::views(bMesh.faceOwners(), bMesh.faceAreas());
    const auto bVsfV = vsf.boundaryData().value().view();
    const auto nAllBoundaryFaces = vsf.boundaryData().value().size();

    NeoN::parallelFor(
        exec,
        {0, nAllBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            const auto own = bOwners[bfi];
            const scalar aSf = bAreas[bfi];
            Kokkos::atomic_add(&numV[own], aSf * Kokkos::max(bVsfV[bfi], lowerBound));
            if (buildSum)
            {
                Kokkos::atomic_add(&denV[own], aSf);
            }
        },
        "bound::averageBoundary"
    );

    // max(max(vsf, average * pos0(-vsf)), lowerBound): the average only replaces cells that are
    // zero or negative; everything else keeps its value and is merely floored.
    NeoN::parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            // Floor the divisor rather than branch on it: the compiler if-converts such a branch
            // into an unconditional division, which would raise FE_INVALID under FOAM_SIGFPE for
            // a zero face-area sum even though the quotient is discarded.
            const scalar avg = numV[celli] / Kokkos::max(denV[celli], NeoN::ROOTVSMALL);
            const scalar refill = vsfV[celli] <= scalar(0) ? avg : scalar(0);
            vsfV[celli] = Kokkos::max(Kokkos::max(vsfV[celli], refill), lowerBound);
        },
        "bound::apply"
    );

    auto bValueV = vsf.boundaryData().value().view();
    NeoN::parallelFor(
        exec,
        {0, nAllBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) { bValueV[bfi] = Kokkos::max(bValueV[bfi], lowerBound); },
        "bound::applyBoundary"
    );

    return true;
}

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
