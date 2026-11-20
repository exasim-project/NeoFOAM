// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/ordering/permutationApplicator.hpp"
#include <stdexcept>

namespace
{
template<typename T>
void reorderCellData(
    const NeoN::Executor& exec,
    const NeoFOAM::PreparedPermutation& permutation,
    NeoN::Vector<T>& data
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
    const NeoFOAM::PreparedPermutation& permutation,
    NeoN::UnstructuredMesh& mesh
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
// --- Definitions of public member functions ---
void PermutationApplicator::apply(
    const PreparedPermutation& permutation,
    NeoN::UnstructuredMesh& mesh
) const
{
    validateCompatibility(permutation, mesh);

    if (!permutation.isIdentity())
    {
        reorderCellData(permutation, mesh.cellVolumes());

        reorderCellData(permutation, mesh.cellCenters());

        remapConnectivity(permutation, mesh);
    }
}

// template<typename ValueType>
// void PermutationApplicator::apply(
//     NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& field,
//     const PreparedPermutation& permutation
// ) const
// {
//     validateCompatibility(field, permutation);
// }

// --- Definitions of private member functions ---
void PermutationApplicator::validateCompatibility(
    const PreparedPermutation& permutation,
    const NeoN::UnstructuredMesh& mesh
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

// template<typename ValueType>
// void validateCompatibility(
//     const NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& field,
//     const PreparedPermutation& permutation
// ) const
// {

// }

template<typename T>
void PermutationApplicator::reorderCellData(
    const PreparedPermutation& permutation,
    NeoN::Vector<T>& data
) const
{
    ::reorderCellData(exec_, permutation, data);
}

void PermutationApplicator::remapConnectivity(
    const PreparedPermutation& permutation,
    NeoN::UnstructuredMesh& mesh
) const
{
    ::remapConnectivity(exec_, permutation, mesh);
}

} // namespace NeoFOAM
