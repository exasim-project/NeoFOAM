// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/auxiliary/bound.hpp"

#include <limits>

namespace NeoFOAM
{

bool bound(
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
    // surfaceSum(magSf*ssf)/surfaceSum(magSf) over internal, boundary AND processor faces.
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
    const auto nBoundaryFaces = mesh.nBoundaryFaces();

    NeoN::parallelFor(
        exec,
        {0, nBoundaryFaces},
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

    // Processor faces sit at the tail of boundaryData().value() starting at nBoundaryFaces and
    // hold the GHOST-CELL value, not a face value. Interpolate owner and ghost with the boundary
    // weight, so the average is the one fvc::average computes and does not depend on the
    // decomposition.
    const auto nProcFaces = mesh.nProcBoundaryFaces();
    if (nProcFaces > 0)
    {
        const auto bWeights = bMesh.weights().view();
        NeoN::parallelFor(
            exec,
            {0, nProcFaces},
            NEON_LAMBDA(const localIdx proci) {
                const auto bfi = nBoundaryFaces + proci;
                const auto own = bOwners[bfi];
                const scalar w = bWeights[bfi];
                const scalar bOwn = Kokkos::max(vsfV[own], lowerBound);
                const scalar bGhost = Kokkos::max(bVsfV[bfi], lowerBound);
                const scalar faceValue = w * bOwn + (scalar(1) - w) * bGhost;
                const scalar aSf = bAreas[bfi];
                Kokkos::atomic_add(&numV[own], aSf * faceValue);
                if (buildSum)
                {
                    Kokkos::atomic_add(&denV[own], aSf);
                }
            },
            "bound::averageProcBoundary"
        );
    }

    // max(max(vsf, average * pos0(-vsf)), lowerBound): the average only replaces cells that are
    // zero or negative; everything else keeps its value and is merely floored.
    // Kokkos::max takes its arguments by const reference, and binding one to the namespace-scope
    // NeoN::ROOTVSMALL would ODR-use a host-only constant from device code ("identifier
    // NeoN::ROOTVSMALL is undefined in device code" under nvcc). Copy it into a local the lambda
    // captures by value instead.
    const scalar rootVSmall = NeoN::ROOTVSMALL;
    NeoN::parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            // Floor the divisor rather than branch on it: the compiler if-converts such a branch
            // into an unconditional division, which would raise FE_INVALID under FOAM_SIGFPE for
            // a zero face-area sum even though the quotient is discarded.
            const scalar avg = numV[celli] / Kokkos::max(denV[celli], rootVSmall);
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

} // namespace NeoFOAM
