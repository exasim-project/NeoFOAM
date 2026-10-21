// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/ordering/permutationApplicator.hpp"
#include <stdexcept>

namespace NeoFOAM
{
PermutationApplicator::PermutationApplicator(const NeoN::Executor& exec)
    : exec_ {exec}
{
}

void PermutationApplicator::apply(
    NeoN::UnstructuredMesh& mesh,
    const PreparedPermutation& permutation) const
{
    // validate permutation
    validate(mesh, permutation);

    // reorder cell volumes
    // reorderCellVolumes(mesh, permutation);

    // reorder cell centers

    // remap face owners

    // remap face neighbors
}

// void PermutationApplicator::apply(
//     NeoN::Field<T>& field,
//     const PreparedPermutation& permutation) const
// {
//     // validate permutation

//     // reorder field values
// }

void PermutationApplicator::validate(
    const NeoN::UnstructuredMesh& mesh,
    const PreparedPermutation& permutation) const
{
    if (permutation.size() != mesh.nCells())
    {
        throw std::invalid_argument(
            "Permutation size does not match number of cells in mesh.");
    }
}


// new cell volumes [oldToNew[old index i]] = old cell volumes [old index i] 
void PermutationApplicator::reorderCellVolumes(
    const NeoN::UnstructuredMesh& mesh,
    const PreparedPermutation& permutation) const
{
    // const NeoN::Executor exec = this->exec();
    // const NeoN::sizeType permutationSize = permutation.size();
    // NeoN::scalarVector reorderedCellVolumes(exec, permutationSize, 0.0);
    // auto reorderedCellVolumesView = reorderedCellVolumes.view();
    // auto originalCellVolumesView = mesh.cellVolumes().view();

    // NeoN::parallelFor(
    //     exec,
    //     {0, NeoN::toLocalIdx(permutationSize)},
    //     NEON_LAMBDA(const NeoN::localIdx i){

    //         // reorderCellVolumes[permutation.oldToNew()[i]] = mesh.cellVolumes()[i];
    //     },
    //     "reorder cell volumes"
    // );
}
} // namespace NeoFOAM


