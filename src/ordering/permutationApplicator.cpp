// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

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
    const Permutation& permutation) const
{
    // validate permutation

    // reoder cell volumes

    // reorder cell centers

    // remap face owners

    // remap face neighbors
}

void PermutationApplicator::apply(
    NeoN::Field<T>& field,
    const Permutation& permutation) const
{
    // validate permutation

    // reorder field values
}

void PermutationApplicator::validate(
    const NeoN::UnstructuredMesh& mesh,
    const Permutation& permutation) const
{
    using SizeType = NeoFOAM::Permutation::SizeType;
    if (permutation.size() != static_cast<SizeType>(mesh.nCells()))
    {
        throw std::invalid_argument(
            "Permutation size does not match number of cells in mesh.");
    }
}
} // namespace NeoFOAM


