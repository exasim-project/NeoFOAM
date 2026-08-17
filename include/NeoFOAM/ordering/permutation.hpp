// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace NeoFOAM
{
/**
 * @brief Represents a bijective mapping between an old and a new index space.
 * 
 * A Permutation is an immutable valule type that describes how a set of entities
 * is renumbered without modifying or reordering the associated data.
 * 
 * For a permutation of size @c N:
 * 
 * - Every old index in [0, N) maps to exactly one new index in [0, N).
 * - Every new index in [0, N) maps to exactly one old index in [0, N).
 * - @c oldToNew and @c newToOld are inverse mappings.
 * 
 * The class provides both mapping directions so that it can be used fo
 * both reordering entitiy-associated data and remapping entity references.
 * 
 * @note Permutation only represents a mapping. It does not compute an ordering
 *       and does not apply the mapping to mesh, field, or other CFD data. Ordering
 *       algorithms produce Permutation objects, while permutation application is
 *       handled separately by the ordering fraemework.
 */
class Permutation
{
public:

    using IndexType = NeoN::label;

    /**
     * @brief Constructs a permutation from an old-to-new mapping.
     *
     * The supplied mapping defines the new index of every entity in the
     * original index space. The corresponding new-to-old mapping is
     * constructed as its inverse.
     *
     * @param oldToNew Mapping from each old index to its corresponding
     *                 new index.
     *
     * @pre @p oldToNew represents a bijection over the index range
     *     [0, oldToNew.size()).
     *
     * @post The resulting permutation contains both the old-to-new
     *       mapping and its inverse.
     */
    explicit Permutation(std::vector<IndexType> oldToNew);


    /**
     * @brief Creates the identity permutation of the specified size.
     *
     * The identity permutation maps every index to itself:
     *
     * @code
     * oldToNew(i) == i
     * newToOld(i) == i
     * @endcode
     *
     * @param size Number of indices represented by the permutation.
     *
     * @return An identity permutation of size @p size.
     */
    [[nodiscard]]
    static Permutation identity(std::size_t size);

    /**
     * @brief Returns the number of indices represented by the permutation.
     */
    [[nodiscard]]
    std::size_t size() const noexcept
    {
        return oldToNew_.size();
    }

    /**
     * @brief Returns the new index corresponding to an old index.
     *
     * @param oldIndex Index in the original index space.
     *
     * @return The corresponding index in the new index space.
     *
     * @pre @p oldIndex is in the range [0, size()).
     */
    [[nodiscard]]
    IndexType oldToNew(IndexType oldIndex) const;

    /**
     * @brief Returns the old index corresponding to a new index.
     *
     * @param newIndex Index in the new index space.
     *
     * @return The corresponding index in the original index space.
     *
     * @pre @p newIndex is in the range [0, size()).
     */
    [[nodiscard]]
    IndexType newToOld(IndexType newIndex) const;

    /**
     * @brief Returns a read-only view of the old-to-new mapping.
     *
     * The returned view contains the new index corresponding to each old
     * index.
     *
     * @return A non-owning read-only view of the old-to-new mapping.
     *
     * @note The returned view refers to storage owned by this Permutation
     *       and is valid only while that storage remains valid.
     */
    [[nodiscard]]
    std::span<const IndexType> oldToNew() const noexcept;

    /**
     * @brief Returns a read-only view of the new-to-old mapping.
     *
     * The returned view contains the old index corresponding to each new
     * index.
     *
     * @return A non-owning read-only view of the new-to-old mapping.
     *
     * @note The returned view refers to storage owned by this Permutation
     *       and is valid only while that storage remains valid.
     */
    [[nodiscard]]
    std::span<const IndexType> newToOld() const noexcept;

    /**
     * @brief Returns the inverse permutation.
     *
     * The inverse permutation exchanges the roles of the old and new
     * index spaces. For a permutation @p p, the inverse satisfies:
     *
     * @code
     * p.inverse().oldToNew(i) == p.newToOld(i)
     * p.inverse().newToOld(i) == p.oldToNew(i)
     * @endcode
     *
     * @return The inverse of this permutation.
     */
    [[nodiscard]]
    Permutation inverse() const;

    // [[nodiscard]]
    // Permutation compose(
    //     const Permutation& other
    // ) const;

    /**
     * @brief Returns whether this permutation is the identity permutation.
     *
     * A permutation is an identity permutation if every index maps to
     * itself.
     *
     * @return @c true if the permutation is an identity permutation;
     *         otherwise, @c false.
     */
    [[nodiscard]]
    bool isIdentity() const noexcept;

private:

    std::vector<IndexType> oldToNew_;
    std::vector<IndexType> newToOld_;
};
} // namespace NeoFOAM