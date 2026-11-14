// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstddef>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "NeoN/NeoN.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"
#endif

#include "NeoFOAM/algorithms/vof/courant.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

namespace
{

using NeoN::localIdx;
using NeoN::scalar;

// Faces of every cell in fvc::surfaceSum's order: internal faces by index (as owner or
// neighbour), then the boundary faces, stored as nInternalFaces + boundary index. A per-cell
// gather in this order adds up exactly as OpenFOAM does, which atomics on threads cannot.
struct CellFaces
{
    NeoN::Vector<localIdx> offsets;
    NeoN::Vector<localIdx> faces;
};

const CellFaces& cellFaces(const NeoN::UnstructuredMesh& mesh)
{
    auto& db = mesh.stencilDB();
    if (!db.contains("VofCellFaces"))
    {
        const auto ownHost = mesh.faceOwners().copyToHost();
        const auto neiHost = mesh.faceNeighbors().copyToHost();
        const auto bOwnHost = mesh.boundaryMesh().faceOwners().copyToHost();
        const auto own = ownHost.view();
        const auto nei = neiHost.view();
        const auto bOwn = bOwnHost.view();
        const localIdx nInternal = mesh.nInternalFaces();
        const auto nBoundary = static_cast<localIdx>(bOwn.size());

        std::vector<localIdx> offsets(static_cast<std::size_t>(mesh.nCells()) + 1, 0);
        for (localIdx f = 0; f < nInternal; ++f)
        {
            ++offsets[own[f] + 1];
            ++offsets[nei[f] + 1];
        }
        for (localIdx bf = 0; bf < nBoundary; ++bf)
        {
            ++offsets[bOwn[bf] + 1];
        }
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());

        std::vector<localIdx> next(offsets.begin(), offsets.end() - 1);
        std::vector<localIdx> faces(offsets.back());
        for (localIdx f = 0; f < nInternal; ++f)
        {
            faces[next[own[f]]++] = f;
            faces[next[nei[f]]++] = f;
        }
        for (localIdx bf = 0; bf < nBoundary; ++bf)
        {
            faces[next[bOwn[bf]]++] = nInternal + bf;
        }
        db.insert(
            "VofCellFaces",
            std::make_shared<CellFaces>(CellFaces {
                NeoN::Vector<localIdx>(mesh.exec(), offsets),
                NeoN::Vector<localIdx>(mesh.exec(), faces)
            })
        );
    }
    return *db.get<std::shared_ptr<CellFaces>>("VofCellFaces");
}

} // namespace

std::pair<scalar, scalar> courantNumbers(
    const fvcc::VolumeField<scalar>& alpha,
    const fvcc::SurfaceField<scalar>& phi,
    scalar deltaT
)
{
    const auto& mesh = alpha.mesh();
    const CellFaces& map = cellFaces(mesh);
    const auto [offsets, faces, V, a, phiI] = NeoN::views(
        map.offsets,
        map.faces,
        mesh.cellVolumes(),
        alpha.internalVector(),
        phi.internalVector()
    );
    const auto phiB = phi.boundaryData().value().view();
    const localIdx nInternal = mesh.nInternalFaces();

    // One pass for both maxima: MinMax carries max(sumPhi/V) and -max(interface sumPhi/V),
    // negation being exact. All terms are >= 0, so {0, 0} is the identity on Serial (which
    // starts from this value; the parallel backends initialise the reducer themselves).
    Kokkos::MinMaxScalar<scalar> extrema {0.0, 0.0};
    Kokkos::MinMax<scalar> reducer(extrema);
    NeoN::parallelReduce(
        alpha.exec(),
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c, Kokkos::MinMaxScalar<scalar>& acc) {
            scalar sumPhi = 0.0;
            for (localIdx i = offsets[c]; i < offsets[c + 1]; ++i)
            {
                const localIdx f = faces[i];
                sumPhi += Kokkos::abs(f < nInternal ? phiI[f] : phiB[f - nInternal]);
            }
            const scalar coCell = sumPhi / V[c];
            const bool nearInterface = a[c] > 0.01 && a[c] < 0.99;
            acc.max_val = Kokkos::max(acc.max_val, coCell);
            acc.min_val = Kokkos::min(acc.min_val, nearInterface ? -coCell : 0.0);
        },
        reducer
    );
    scalar maxima[2] = {extrema.max_val, 0.0 - extrema.min_val};

#ifdef NF_WITH_MPI_SUPPORT
    if (mesh.boundaryMesh().isDistributed())
    {
        NeoN::mpi::Environment env;
        MPI_Allreduce(MPI_IN_PLACE, maxima, 2, NeoN::mpi::getType<scalar>(), MPI_MAX, env.comm());
    }
#endif

    // CourantNo.H's operation order: 0.5*gMax(...)*deltaT.
    return {0.5 * maxima[0] * deltaT, 0.5 * maxima[1] * deltaT};
}

} // namespace NeoFOAM
