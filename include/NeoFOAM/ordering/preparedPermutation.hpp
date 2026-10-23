// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"
#include "NeoFOAM/ordering/permutation.hpp"

namespace NeoFOAM
{
class PreparedPermutation
{
public:
    using IndexType = NeoN::label;
    /**
     * @brief Prepare a permutation for execution on the given executor.
     *  
     * @param permutation Canonical host-resident permutation,
     * @param exec Executor whose memory space will store the mappings.
     */
    explicit PreparedPermutation(
        const NeoN::Executor& exec,
        const Permutation& permutation
    )
        : oldToNew_(
            exec, 
            permutation.oldToNew().data(),
            permutation.size()
        ),
        newToOld_(
            exec,
            permutation.newToOld().data(),
            permutation.size()
        )
    {}
    
    /**
     * @brief Return the old-to-new mapping.
     * 
     * The returned view is non-owning and read-only
     */
    [[nodiscard]]
    NeoN::View<const IndexType> oldToNew() const
    {
        return oldToNew_.view();
    }

    /**
     * @brief Return the new-to-old mapping.
     * 
     * The returned view is non-owning and read-only.
     */
    [[nodiscard]]
    NeoN::View<const IndexType> newToOld() const
    {
        return newToOld_.view();
    }

    /**
     * @brief Return the number of entities in the permutation.
     */
    [[nodiscard]]
    NeoN::sizeType size() const noexcept
    {
        return oldToNew_.size();
    }

    /**
     * @brief Return the executor used to prepare the permutation.
     */
    [[nodiscard]]
    const NeoN::Executor& exec() const noexcept
    {
        return oldToNew_.exec();
    }

private:
    NeoN::Array<IndexType> oldToNew_;
    NeoN::Array<IndexType> newToOld_;
}; 
} // namespace NeoFOAM