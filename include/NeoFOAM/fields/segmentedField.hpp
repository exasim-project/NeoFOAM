// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/field.hpp"

namespace NeoFOAM
{

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
    std::pair<std::span<ValueType>, std::span<IndexType>> spans()
    {
        return {values.span(), segments.span()};
    }
};

} // namespace NeoFOAM
