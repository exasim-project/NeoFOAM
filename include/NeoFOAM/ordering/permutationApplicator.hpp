// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"
#include "NeoFOAM/ordering/preparedPermutation.hpp"

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
 * @note The applicator does not:
 *       - compute permutations
 *       - modify permutation objects
 *       - select ordering algorithms
 *       - perform implicit host/device data transfers
 */
class PermutationApplicator
{
public:

    /**
     * @brief Constructs a permutation applicator for the specified executor.
     *
     * @param exec Executor used to perform permutation application operations.
     */
    explicit PermutationApplicator(const NeoN::Executor& exec)
        : exec_ {exec}
    {}

    // return executor on which the applicator is performed
    [[nodiscard]] const NeoN::Executor& exec() const noexcept { return exec_; }

    /**
     * @brief Applies a cell permutation to a mesh.
     *
     * Cell-associated mesh data is reordered according to @p permutation,
     * while cell indices stored in mesh connectivity are remapped to their
     * corresponding new indices.
     *
     * In particular, cell-associated mesh data is reordered by position, whereas
     * face owner and neighbour indices are remapped by value.
     *
     * @param permutation Permutation defining the mapping from old cell
     * @param mesh Mesh whose cell-related mesh data is to be reordered.
     *                    indices to new cell indices.
     *
     * @pre @p permutation is a valid permutation over the cells of @p mesh.
     *
     * @post Cell-associated mesh data and cell references are consistent with
     *       the new cell ordering.
     *
     * @throws std::invalid_argument exception if the size of
     *         @p permutation does not match the number of cells of
     *         @p mesh.
     *
     * @note Mesh faces are not reordered by this function.
     */
    void apply(const PreparedPermutation& permutation, NeoN::UnstructuredMesh& mesh) const;

    /**
     * @brief Applies a cell permutation to a cell-centred volume field.
     *
     * The internal (cell) values of @p field are reordered by position
     * according to @p permutation, so that the value stored for old cell
     * index @c i is moved to the corresponding new cell index.
     *
     * Boundary data is left unchanged: a cell permutation does not change the
     * ordering of boundary faces.
     *
     * This function reorders only the field. It does not modify the mesh
     * associated with @p field. To keep the mesh and its fields consistent,
     * the caller must apply the same permutation to the mesh and to every
     * cell-centred field defined on it, exactly once each.
     *
     * @tparam ValueType Value type of the field (e.g. scalar, Vec3).
     *
     * @param permutation Permutation defining the mapping from old cell
     *                    indices to new cell indices.
     * @param field Volume field whose internal values are to be reordered.
     *
     * @pre @p permutation is a valid permutation over the cells of the mesh
     *      on which @p field is defined, i.e. its size equals the number of
     *      internal values of @p field.
     * @pre @p field has not already been reordered with @p permutation.
     *
     * @post The internal values of @p field are in the new cell ordering.
     *       Boundary data of @p field is unchanged.
     *
     * @throws std::invalid_argument exception if the size of
     *         @p permutation does not match the number of internal values of
     *         @p field.
     *
     * @note Surface (face-centred) fields are not handled by this function.
     *       A cell permutation does not reorder faces.
     */
    // template<typename ValueType>
    // void apply(
    //     NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& field,
    //     const PreparedPermutation& permutation
    // ) const;


private:

    NeoN::Executor exec_;

    void validateCompatibility(
        const PreparedPermutation& permutation,
        const NeoN::UnstructuredMesh& mesh
    ) const;

    // template<typename ValueType>
    // void validateCompatibility(
    //     const NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& field,
    //     const PreparedPermutation& permutation
    // ) const;

    template<typename T>
    void reorderCellData(const PreparedPermutation& permutation, NeoN::Vector<T>& data) const;

    void
    remapConnectivity(const PreparedPermutation& permutation, NeoN::UnstructuredMesh& mesh) const;
};

} // namespace NeoFOAM
