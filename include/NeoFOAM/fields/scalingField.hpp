// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

namespace NeoFOAM
{

/**
 * @brief A wrapper class that represents either a span of values or a single value.
 *
 * This class is used to store either a span of values or a single value of type `ValueType`.
 * It provides an indexing operator `operator[]` that returns the value at the specified index
 * or a constant value and is used to scale fields in the DSL.
 *
 * @tparam ValueType The type of the values stored in the class.
 */
template<typename ValueType>
struct ScalingField
{
    ScalingField() = default;
    ScalingField(std::span<ValueType> values) : values(values), value(1.0), useSpan(true) {}

    ScalingField(std::span<ValueType> values, NeoFOAM::scalar value)
        : values(values), value(value), useSpan(true)
    {}

    ScalingField(NeoFOAM::scalar value) : values(), value(value), useSpan(false) {}

    std::span<ValueType> values;
    NeoFOAM::scalar value;
    bool useSpan;

    KOKKOS_INLINE_FUNCTION
    ValueType operator[](const size_t i) const { return useSpan ? values[i] * value : value; }

    void operator*=(NeoFOAM::scalar scale) { value *= scale; }

    void operator=(NeoFOAM::scalar scale) { value = scale; }

    void operator*=(Field<ValueType> scalingField)
    {
        auto scale = scalingField.span();
        // otherwise we are unable to capture values in the lambda
        auto selfValue = values;
        NeoFOAM::parallelFor(
            scalingField.exec(),
            {0, scalingField.size()},
            KOKKOS_LAMBDA(const size_t i) { selfValue[i] *= scale[i]; }
        );
    }
};

} // namespace NeoFOAM
