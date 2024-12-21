// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/field.hpp"

namespace NeoFOAM
{

/**
 * @brief A class representing a segment of indices.
 *
 * @tparam IndexType The type of the indices.
 */
template<typename IndexType>
class Segment
{
public:

    /**
     * @brief A span of indices representing the segments.
     */
    std::span<IndexType> segments;

    /**
     * @brief Get the bounds of a segment.
     *
     * @param segI The index of the segment.
     * @return A pair of indices representing the start and end of the segment.
     */
    KOKKOS_INLINE_FUNCTION
    Kokkos::pair<IndexType, IndexType> bounds(std::size_t segI) const
    {
        return Kokkos::pair<IndexType, IndexType> {segments[segI], segments[segI + 1]};
    }

    /**
     * @brief Get the range of a segment.
     *
     * @param segI The index of the segment.
     * @return A pair of indices representing the start and length of the segment.
     */
    KOKKOS_INLINE_FUNCTION
    Kokkos::pair<IndexType, IndexType> range(std::size_t segI) const
    {
        return Kokkos::pair<IndexType, IndexType> {
            segments[segI], segments[segI + 1] - segments[segI]
        };
    }

    /**
     * @brief Get a subspan of values corresponding to a segment.
     *
     * @tparam ValueType The type of the values.
     * @param values A span of values.
     * @param segI The index of the segment.
     * @return A subspan of values corresponding to the segment.
     */
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION std::span<ValueType>
    subspan(std::span<ValueType> values, std::size_t segI) const
    {
        auto [start, length] = range(segI);
        return values.subspan(start, length);
    }

    /**
     * @brief Access an element of the segments.
     *
     * @param i The index of the element.
     * @return The value of the element at the specified index.
     */
    KOKKOS_INLINE_FUNCTION
    IndexType operator[](std::size_t i) const { return segments[i]; }
};

/**
 * @class SegmentedField
 * @brief Data structure that stores a segmented fields or a vector of vectors
 *
 * @ingroup Fields
 */
template<typename ValueType, typename IndexType>
class SegmentedField
{
public:

    Field<ValueType> values;
    Field<IndexType> segments;

    /**
     * @brief Create a segmented field with a given size and number of segments.
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param numSegments  number of segments
     */
    SegmentedField(const Executor& exec, size_t size, size_t numSegments)
        : values(exec, size), segments(exec, numSegments + 1)
    {}

    /**
     * @brief Create a segmented field from two fields.
     * @param values The values of the segmented field.
     * @param segments The segments of the segmented field.
     */
    SegmentedField(const Field<ValueType>& values, const Field<IndexType>& segments)
        : values(values), segments(segments)
    {}


    /**
     * @brief Get the executor associated with the segmented field.
     * @return Reference to the executor.
     */
    const Executor& exec() const { return values.exec(); }

    /**
     * @brief Get the size of the segmented field.
     * @return The size of the segmented field.
     */
    size_t size() const { return values.size(); }

    /**
     * @brief Get the number of segments in the segmented field.
     * @return The number of segments.
     */
    size_t numSegments() const { return segments.size() - 1; }

    /**
     * @brief get the spans of the segmented field
     * @return Span of the fields
     */
    std::pair<std::span<ValueType>, Segment<IndexType>> spans()
    {
        return {values.span(), Segment<IndexType>(segments.span())};
    }
};

} // namespace NeoFOAM
