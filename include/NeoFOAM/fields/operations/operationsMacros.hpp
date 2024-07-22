// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>

#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/core/types.hpp"
#include "NeoFOAM/helpers/exceptions.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM
{

// Forward declaration
template<StorageType T>
class Field;


template<StorageType T, StorageType Inner>
void map(Field<T>& a, const Inner inner)
{
    parallelFor(a, inner);
}

template<StorageType T>
void fill(Field<T>& a, const std::type_identity_t<T> value)
{
    parallelFor(
        a, KOKKOS_LAMBDA(const size_t) { return value; }
    );
}


template<StorageType T>
void setField(Field<T>& a, const std::span<const std::type_identity_t<T>> b)
{
    parallelFor(
        a, KOKKOS_LAMBDA(const size_t i) { return b[i]; }
    );
}

template<StorageType T>
void scalarMul(Field<T>& a, const std::type_identity_t<T> value)
{
    auto spanA = a.span();
    parallelFor(
        a, KOKKOS_LAMBDA(const size_t i) { return spanA[i] * value; }
    );
}

namespace detail
{
template<StorageType T, typename BinaryOp>
void fieldBinaryOp(Field<T>& a, const Field<std::type_identity_t<T>>& b, BinaryOp op)
{
    NeoFOAM_ASSERT_EQUAL_LENGTH(a, b);
    auto spanA = a.span();
    auto spanB = b.span();
    parallelFor(
        a, KOKKOS_LAMBDA(const size_t i) { return op(spanA[i], spanB[i]); }
    );
}
}

template<StorageType T>
void add(Field<T>& a, const Field<std::type_identity_t<T>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(T va, T vb) { return va + vb; }
    );
}


template<StorageType T>
void sub(Field<T>& a, const Field<std::type_identity_t<T>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(T va, T vb) { return va - vb; }
    );
}

template<StorageType T>
void mul(Field<T>& a, const Field<std::type_identity_t<T>>& b)
{
    detail::fieldBinaryOp(
        a, b, KOKKOS_LAMBDA(T va, T vb) { return va * vb; }
    );
}


} // namespace NeoFOAM
