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

    // reorder cell centers

    // remap face owners

    // remap face neighbors
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
void PermutationApplicator::reorderCellData(
    NeoN::scalarVector& data,
    const PreparedPermutation& permutation) const
{
    const NeoN::sizeType permutationSize = permutation.size();

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
        {0, NeoN::toLocalIdx(permutationSize)},
        NEON_LAMBDA(NeoN::localIdx i){
            auto j = oldToNew[i];
            dataView[j] = oldDataView[i];
        },
        "reorder cell-associated data"
    );
}

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


} // namespace NeoFOAM


