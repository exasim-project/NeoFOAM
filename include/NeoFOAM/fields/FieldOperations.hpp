// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include "NeoFOAM/helpers/exceptions.hpp"

namespace NeoFOAM
{


#define DECLARE_UNARY_FIELD_OP(Name, Kernel_Impl)                             \
    template<typename T, typename Inner>                                      \
    struct Name##Op                                                           \
    {                                                                         \
        template<typename Executor>                                           \
        void                                                                  \
        operator()(const Executor& exec, Field<T>& a, const Inner in)         \
        {                                                                     \
            using executor = typename Executor::exec;                         \
            auto a_f = a.field();                                             \
            Kokkos::parallel_for(                                             \
                Kokkos::RangePolicy<executor>(0, a_f.size()),                 \
                KOKKOS_CLASS_LAMBDA(const int i) { a_f[i] = Kernel_Impl; }    \
            );                                                                \
        }                                                                     \
                                                                              \
        template<typename Executor>                                           \
        void operator()(const CPUExecutor& exec, Field<T>& a, const Inner in) \
        {                                                                     \
            auto a_f = a.field();                                             \
            for (int i = 0; i < a_f.size(); i++)                              \
            {                                                                 \
                a_f[i] = Kernel_Impl;                                         \
            }                                                                 \
        }                                                                     \
    };                                                                        \
    template<typename T, typename Inner>                                      \
    void Name(Field<T>& a, const Inner inner)                                 \
    {                                                                         \
        Name##Op<T, Inner> op_;                                               \
        std::visit([&](const auto& exec) { op_(exec, a, inner); }, a.exec()); \
    }

DECLARE_UNARY_FIELD_OP(map, in(i));
DECLARE_UNARY_FIELD_OP(fill, in);
DECLARE_UNARY_FIELD_OP(setField, in[i]);
DECLARE_UNARY_FIELD_OP(scalar_mul, a_f[i] *= in);

#undef DECLARE_UNARY_FIELD_OP

#define DECLARE_BINARY_FIELD_OP(Name, Kernel_Impl)                               \
    template<typename T>                                                         \
    struct Name##Op                                                              \
    {                                                                            \
        template<typename Executor>                                              \
        void                                                                     \
        operator()(const Executor& exec, Field<T>& a, const Field<T>& b)         \
        {                                                                        \
            using executor = typename Executor::exec;                            \
            auto a_f = a.field();                                                \
            auto b_f = b.field();                                                \
            Kokkos::parallel_for(                                                \
                Kokkos::RangePolicy<executor>(0, a_f.size()),                    \
                KOKKOS_CLASS_LAMBDA(const int i) { a_f[i] = Kernel_Impl; }       \
            );                                                                   \
        }                                                                        \
                                                                                 \
        template<typename Executor>                                              \
        void operator()(const CPUExecutor& exec, Field<T>& a, const Field<T>& b) \
        {                                                                        \
            auto a_f = a.field();                                                \
            const auto b_f = b.field();                                          \
            for (int i = 0; i < a_f.size(); i++)                                 \
            {                                                                    \
                a_f[i] = Kernel_Impl;                                            \
            }                                                                    \
        }                                                                        \
    };                                                                           \
    template<typename T>                                                         \
    void Name(Field<T>& a, const Field<T>& b)                                    \
    {                                                                            \
        NeoFOAM_ASSERT_EQUAL_LENGTH(a, b);                                       \
        Name##Op<T> op_;                                                         \
        std::visit([&](const auto& exec) { op_(exec, a, b); }, a.exec());        \
    }

DECLARE_BINARY_FIELD_OP(add, a_f[i] + b_f[i]);
DECLARE_BINARY_FIELD_OP(sub, a_f[i] - b_f[i]);
DECLARE_BINARY_FIELD_OP(mul, a_f[i] * b_f[i]);

#undef DECLARE_BINARY_FIELD_OP


} // namespace NeoFOAM
