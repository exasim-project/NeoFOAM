// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <tuple>

#include <Kokkos_Core.hpp>
#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/helpers/exceptions.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM
{

// Forward declaration
template<typename ValueType>
class Field;


template<typename T, typename Inner>
void map(Field<T>& field, const Inner inner)
{
    parallelFor(field, inner);
}

template<typename T, typename Inner>
void map(const NeoFOAM::Executor& exec, std::span<T> s, const Inner inner)
{
    parallelFor(exec, s, inner);
}

template<typename ValueType>
void fill(Field<ValueType>& a, const std::type_identity_t<ValueType> value)
{
    parallelFor(
        a, KOKKOS_LAMBDA(const size_t) { return value; }
    );
}


template<typename ValueType>
void setField(Field<ValueType>& a, const std::span<const std::type_identity_t<ValueType>> b)
{
    parallelFor(
        a, KOKKOS_LAMBDA(const size_t i) { return b[i]; }
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

} // namespace NeoFOAM
