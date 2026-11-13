// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/ordering/permutationApplicator.hpp"
#include <stdexcept>

namespace NeoFOAM
{
PermutationApplicator::PermutationApplicator(const NeoN::Executor& exec)
    : exec_ {exec}
{}

void PermutationApplicator::apply(
    NeoN::UnstructuredMesh& mesh,
    const PreparedPermutation& permutation
) const
{
    // validate permutation
    validate(mesh, permutation);

    // Check if the permutation is identity
    if (!permutation.isIdentity())
    {
        // reorder cell volumes
        reorderCellData(mesh.cellVolumes(), permutation);

        // reorder cell centers
        reorderCellData(mesh.cellCenters(), permutation);

        // remap face owners

        // remap face neighbors
    }
}

// template<typename T>
// void PermutationApplicator::apply(
//     NeoN::Field<T>& field,
//     const PreparedPermutation& permutation) const
// {
//     // validate permutation

//     // reorder field values
// }

// new cell volumes [oldToNew[old index i]] = old cell volumes [old index i]
template<typename T>
void PermutationApplicator::reorderCellData(
    NeoN::Vector<T>& data,
    const PreparedPermutation& permutation
) const
{
    // Create a copy of the data with the name old data
    auto oldData = data.makeCopy();

    // Prepare a View of permutation for parallel construct
    const auto oldToNew = permutation.oldToNew();

    // Prepare a View of data for parallel construct
    auto oldDataView = oldData.view();
    auto dataView = data.view();

    // for each old index i,
    // get the new index j
    // data[j] = old data[i]
    NeoN::parallelFor(
        exec_,
        {0, NeoN::toLocalIdx(permutation.size())},
        NEON_LAMBDA(NeoN::localIdx i) {
            auto j = oldToNew[i];
            dataView[j] = oldDataView[i];
        },
        "reorder cell-associated data"
    );
}

// Explicit instantiation of reorderCellData template function
template void PermutationApplicator::reorderCellData<
    NeoN::scalar>(NeoN::Vector<NeoN::scalar>&, const PreparedPermutation&) const;

template void PermutationApplicator::reorderCellData<
    NeoN::Vec3>(NeoN::Vector<NeoN::Vec3>&, const PreparedPermutation&) const;


void PermutationApplicator::remapConnectivity(
    NeoN::UnstructuredMesh& mesh,
    const PreparedPermutation& permutation
) const
{
    // Prepare a view of face owners for parallel construct
    auto faceOwnersView = mesh.faceOwners().view();

    // Prepare a view of oldToNew permutation for parallel construct
    const auto oldToNew = permutation.oldToNew();

    // Remap face owners of interior faces
    NeoN::parallelFor(
        exec_,
        {0, NeoN::toLocalIdx(mesh.nInternalFaces())},
        NEON_LAMBDA(NeoN::localIdx i) {
            auto oldOwnerIdx = faceOwnersView[i];
            faceOwnersView[i] = oldToNew[oldOwnerIdx];
        },
        "remap face owners of internal faces"
    );

    // Remap face neighbors of interior faces
    // Prepare a view of face neighbors for parallel construct
    auto faceNeighborsView = mesh.faceNeighbors().view();
    NeoN::parallelFor(
        exec_,
        {0, NeoN::toLocalIdx(mesh.nInternalFaces())},
        NEON_LAMBDA(NeoN::localIdx i) {
            auto oldNeighborIdx = faceNeighborsView[i];
            faceNeighborsView[i] = oldToNew[oldNeighborIdx];
        },
        "remap face neighbors of internal faces"
    );

    // Remap face owners of boundary faces
    // Prepare a view of face owners for parallel construct
    auto boundaryFaceOwnersView = mesh.boundaryMesh().faceOwners().view();

    NeoN::parallelFor(
        exec_,
        {0, NeoN::toLocalIdx(mesh.nBoundaryFaces())},
        NEON_LAMBDA(NeoN::localIdx i) {
            auto oldOwnerIdx = boundaryFaceOwnersView[i];
            boundaryFaceOwnersView[i] = oldToNew[oldOwnerIdx];
        },
        "remap face owners of boundary faces"
    );

    // Correct the face owner and neighbor indices according to OpenFOAM convention
}

void PermutationApplicator::validate(
    const NeoN::UnstructuredMesh& mesh,
    const PreparedPermutation& permutation
) const
{
    if (permutation.size() != mesh.nCells())
    {
        throw std::invalid_argument("Permutation size does not match number of cells in mesh.");
    }
}
} // namespace NeoFOAM
