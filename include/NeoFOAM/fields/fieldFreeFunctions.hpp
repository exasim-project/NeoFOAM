// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <tuple>
#include <span>

#include <Kokkos_Core.hpp>
#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/helpers/exceptions.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM
{

// Forward declaration
template<typename ValueType>
class Field;


/**
 * @brief Map a field using a specific executor.
 *
 * @param a The field to map.
 * @param inner The function to apply to each element of the field.
 * @param range The range to map the field in. If not provided, the whole field is mapped.
 */
template<typename T, typename Inner>
void map(Field<T>& a, const Inner inner, std::pair<size_t, size_t> range = {0, 0})
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = a.size();
    }
    auto spanA = a.span();
    parallelFor(
        a.exec(), {start, end}, KOKKOS_LAMBDA(const size_t i) { spanA[i] = inner(i); }
    );
}

/**
 * @brief Fill the field with a scalar value using a specific executor.
 *
 * @param field The field to fill.
 * @param value The scalar value to fill the field with.
 * @param range The range to fill the field in. If not provided, the whole field is filled.
 */
template<typename ValueType>
void fill(
    Field<ValueType>& a,
    const std::type_identity_t<ValueType> value,
    std::pair<size_t, size_t> range = {0, 0}
)
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = a.size();
    }
    auto spanA = a.span();
    parallelFor(
        a.exec(), {start, end}, KOKKOS_LAMBDA(const size_t i) { spanA[i] = value; }
    );
}


/**
 * @brief Set the field with a span of values using a specific executor.
 *
 * @param a The field to set.
 * @param b The span of values to set the field with.
 * @param range The range to set the field in. If not provided, the whole field is set.
 */
template<typename ValueType>
void setField(
    Field<ValueType>& a,
    const std::span<const std::type_identity_t<ValueType>> b,
    std::pair<size_t, size_t> range = {0, 0}
)
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = a.size();
    }
    auto spanA = a.span();
    parallelFor(
        a.exec(), {start, end}, KOKKOS_LAMBDA(const size_t i) { spanA[i] = b[i]; }
    );
}

template<typename ValueType>
void scalarMul(Field<ValueType>& a, const std::type_identity_t<ValueType> value)
{
    auto spanA = a.span();
    parallelFor(
        a, KOKKOS_LAMBDA(const size_t i) { return spanA[i] * value; }
    );
}

namespace detail
{
template<typename ValueType, typename BinaryOp>
void fieldBinaryOp(
    Field<ValueType>& a, const Field<std::type_identity_t<ValueType>>& b, BinaryOp op
)
{
    NeoFOAM_ASSERT_EQUAL_LENGTH(a, b);
    auto spanA = a.span();
    auto spanB = b.span();
    parallelFor(
        a, KOKKOS_LAMBDA(const size_t i) { return op(spanA[i], spanB[i]); }
    );
}
}

template<typename ValueType>
void add(Field<ValueType>& a, const Field<std::type_identity_t<ValueType>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va + vb; }
    );
}


template<typename ValueType>
void sub(Field<ValueType>& a, const Field<std::type_identity_t<ValueType>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va - vb; }
    );
}

template<typename ValueType>
void mul(Field<ValueType>& a, const Field<std::type_identity_t<ValueType>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va * vb; }
    );
}

template<typename... Args>
auto spans(Args&... fields)
{
    return std::make_tuple(fields.span()...);
}

template<typename... Args>
auto copyToHosts(Args&... fields)
{
    return std::make_tuple(fields.copyToHost()...);
}

template<typename T>
bool equal(Field<T>& field, T value)
{
    auto hostField = field.copyToHost();
    auto hostSpan = hostField.span();
    for (size_t i = 0; i < hostSpan.size(); i++)
    {
        if (hostSpan[i] != value)
        {
            return false;
        }
    }
    return true;
};

template<typename T>
bool equal(const Field<T>& field, const Field<T>& field2)
{
    auto [hostField, hostField2] = copyToHosts(field, field2);
    auto [hostSpan, hostSpan2] = spans(hostField, hostField2);

    if (hostSpan.size() != hostSpan2.size())
    {
        return false;
    }

    for (size_t i = 0; i < hostSpan.size(); i++)
    {
        if (hostSpan[i] != hostSpan2[i])
        {
            return false;
        }
    }

    return true;
};

template<typename T>
bool equal(const Field<T>& field, std::span<T> span2)
{
    auto hostSpan = field.copyToHost().span();

    if (hostSpan.size() != span2.size())
    {
        return false;
    }

    for (size_t i = 0; i < hostSpan.size(); i++)
    {
        if (hostSpan[i] != span2[i])
        {
            return false;
        }
    }

    return true;
}

} // namespace NeoFOAM
