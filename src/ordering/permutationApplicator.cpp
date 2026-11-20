// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/ordering/permutationApplicator.hpp"
#include <stdexcept>

namespace
{
template<typename T>
void reorderCellData(
    const NeoN::Executor& exec,
    NeoN::Vector<T>& data,
    const NeoFOAM::PreparedPermutation& permutation
)
{
    auto oldData = data.makeCopy();

    const auto oldToNew = permutation.oldToNew();

    auto oldDataView = oldData.view();
    auto dataView = data.view();

    NeoN::parallelFor(
        exec,
        {0, NeoN::toLocalIdx(permutation.size())},
        NEON_LAMBDA(NeoN::localIdx i) {
            auto j = oldToNew[i];
            dataView[j] = oldDataView[i];
        },
        "reorder cell-associated data"
    );
}

void remapConnectivity(
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const NeoFOAM::PreparedPermutation& permutation
)
{
    auto faceOwnersView = mesh.faceOwners().view();

    const auto oldToNew = permutation.oldToNew();

    // Remap face owners of interior faces
    NeoN::parallelFor(
        exec,
        {0, NeoN::toLocalIdx(mesh.nInternalFaces())},
        NEON_LAMBDA(NeoN::localIdx i) {
            auto oldOwnerIdx = faceOwnersView[i];
            faceOwnersView[i] = oldToNew[oldOwnerIdx];
        },
        "remap face owners of internal faces"
    );

    // Remap face neighbors of interior faces
    auto faceNeighborsView = mesh.faceNeighbors().view();

    NeoN::parallelFor(
        exec,
        {0, NeoN::toLocalIdx(mesh.nInternalFaces())},
        NEON_LAMBDA(NeoN::localIdx i) {
            auto oldNeighborIdx = faceNeighborsView[i];
            faceNeighborsView[i] = oldToNew[oldNeighborIdx];
        },
        "remap face neighbors of internal faces"
    );

    // Remap face owners of boundary faces
    auto boundaryFaceOwnersView = mesh.boundaryMesh().faceOwners().view();

    NeoN::parallelFor(
        exec,
        {0, NeoN::toLocalIdx(mesh.nBoundaryFaces())},
        NEON_LAMBDA(NeoN::localIdx i) {
            auto oldOwnerIdx = boundaryFaceOwnersView[i];
            boundaryFaceOwnersView[i] = oldToNew[oldOwnerIdx];
        },
        "remap face owners of boundary faces"
    );

    // Correct the face owner and neighbor indices according to OpenFOAM convention
    NeoN::parallelFor(
        exec,
        {0, NeoN::toLocalIdx(mesh.nInternalFaces())},
        NEON_LAMBDA(NeoN::localIdx i) {
            auto ownerIdx = faceOwnersView[i];
            auto neighborIdx = faceNeighborsView[i];
            if (ownerIdx > neighborIdx)
            {
                faceOwnersView[i] = neighborIdx;
                faceNeighborsView[i] = ownerIdx;
            }
        },
        "correct face owner and neighbor indices of internal faces"
    );
}
} // anonymous namespace

namespace NeoFOAM
{
// PermutationApplicator::PermutationApplicator(const NeoN::Executor& exec)
//     : exec_ {exec}
// {}

void PermutationApplicator::apply(
    NeoN::UnstructuredMesh& mesh,
    const PreparedPermutation& permutation
) const
{
    validateCompatibility(mesh, permutation);

    if (!permutation.isIdentity())
    {
        reorderCellData(mesh.cellVolumes(), permutation);

        reorderCellData(mesh.cellCenters(), permutation);

        remapConnectivity(mesh, permutation);
    }
}


void PermutationApplicator::validateCompatibility(
    const NeoN::UnstructuredMesh& mesh,
    const PreparedPermutation& permutation
) const
{
    if (permutation.size() != mesh.nCells())
    {
        throw std::invalid_argument(
            "The permutation size (" + std::to_string(permutation.size())
            + ") does not match the number of cells in the mesh (" + std::to_string(mesh.nCells())
            + ")."
        );
    }
}

template<typename T>
void PermutationApplicator::reorderCellData(
    NeoN::Vector<T>& data,
    const PreparedPermutation& permutation
) const
{
    ::reorderCellData(exec_, data, permutation);
}

void PermutationApplicator::remapConnectivity(
    NeoN::UnstructuredMesh& mesh,
    const PreparedPermutation& permutation
) const
{
    ::remapConnectivity(exec_, mesh, permutation);
}

} // namespace NeoFOAM
