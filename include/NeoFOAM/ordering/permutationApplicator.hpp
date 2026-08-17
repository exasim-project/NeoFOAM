// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"
#include "NeoFOAM/ordering/permutation.hpp"

namespace NeoFOAM
{
/**
 * @brief Applies cell permutations to mesh and cell-associated data.
 * 
 * PermutationApplicator applies a valid cell permutation to NeoN mesh data
 * while preserving the consistency of cell-associated data and references.
 * 
 * A cell permutation has two distinct effects:
 * 
 * - Cell-associated data, such as cell fields and cell geometry, is reordered
 *   according to the permutation.
 * - Cell references stored in mesh connectivity, such as face owner and 
 *   neighbor indices, are remapped according to the permutation.
 * 
 * The aplicator does not:
 * - compute permutations
 * - modify permutation objects
 * - select ordering algorithms
 * - perform implicit host/device data transfers
 */
class PermutationApplicator
{
public:

    /**
     * @brief Constructs a permutation applicator for the specified executor.
     *
     * @param exec Executor used to perform permutation application operations.
     */
    explicit PermutationApplicator(const NeoN::Executor& exec);

    /**
     * @brief Applies a cell permutation to a mesh.
     *
     * Cell-associated mesh data is reordered according to @p permutation,
     * while cell indices stored in mesh connectivity are remapped to their
     * corresponding new indices.
     *
     * In particular, cell-associated data is reordered by position, whereas
     * face owner and neighbour indices are remapped by value.
     *
     * @param mesh Mesh whose cell-related data is to be reordered.
     * @param permutation Permutation defining the mapping from old cell
     *                    indices to new cell indices.
     *
     * @pre @p permutation is a valid permutation over the cells of @p mesh.
     *
     * @post Cell-associated mesh data and cell references are consistent with
     *       the new cell ordering.
     */
    void apply(
        NeoN::UnstructuredMesh& mesh,
        const Permutation& permutation) const;
     
    /**
     * @brief Applies a cell permutation to a cell-associated field.
     *
     * The values of @p field are reordered according to @p permutation so
     * that each value remains associated with the same physical cell after
     * the cell ordering changes.
     *
     * @tparam T Value type of the cell-associated field.
     *
     * @param field Cell-associated field to reorder.
     * @param permutation Permutation defining the mapping from old cell
     *                    indices to new cell indices.
     *
     * @pre @p permutation is a valid permutation over the cells represented
     *      by @p field.
     *
     * @post Each field value remains associated with the same cell as before
     *       the permutation was applied.
     */
    template<typename T>
    void apply(
        NeoN::Field<T>& field,
        const Permutation& permutation) const;

private:
    NeoN::Executor exec_;
};
} // namespace NeoFOAM