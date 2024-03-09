// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include "NeoFOAM/helpers/exceptions.hpp"

namespace NeoFOAM
{


template<typename T>
struct UnaryKernelFunctor
{
    template<typename Executor, typename KernelImplementation>
    void
    operator()(const Executor& exec, Field<T>& a, const KernelImplementation f)
    {
        using executor = typename Executor::exec;
        auto a_f = a.field();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<executor>(0, a_f.size()),
            KOKKOS_CLASS_LAMBDA(const int i) { a_f[i] = f(i); }
        );
    }

    template<typename Executor, typename KernelImplementation>
    void operator()(const CPUExecutor& exec, Field<T>& a, const Inner in)
    {
        auto a_f = a.field();
        for (int i = 0; i < a_f.size(); i++)
        {
            a_f[i] = KernelImplementation(i);
        }
    }
};

template<typename T, typename Inner>
void map(Field<T>& a, const Inner inner)
{
    UnaryKernelFunctor<T, Inner> op;
    std::visit([&](const auto& exec)
               { op(exec, a, [](const int i)
                    { return inner(i); }); },
               a.exec());
}

template<typename T, typename Inner>
void fill(Field<T>& a, const Inner inner)
{
    UnaryKernelFunctor<T, Inner> op;
    std::visit([&](const auto& exec)
               { op(exec, a, [](const int i)
                    { return inner; }); },
               a.exec());
}

template<typename T, typename Inner>
void setField(Field<T>& a, const Inner inner)
{
    UnaryKernelFunctor<T, Inner> op;
    std::visit([&](const auto& exec)
               { op(exec, a, [](const int i)
                    { return inner[i]; }); },
               a.exec());
}

template<typename T, typename Inner>
void scalar_mul(Field<T>& a, const Inner inner)
{
    // FIXME
    UnaryKernelFunctor<T, Inner> op;
    std::visit([&](const auto& exec)
               { op(exec, a, [](const int i)
                    { return inner[i]; }); },
               a.exec());
}

// UNARY_FIELD_OP(map, in(i));
// UNARY_FIELD_OP(fill, in);
// UNARY_FIELD_OP(setField, in[i]);
// UNARY_FIELD_OP(scalar_mul, a_f[i] *= in);

#undef UNARY_FIELD_OP

// NOTE DONT MERGE do we need the CPU version of this here, should we call this a reference
// implementation?
#define BINARY_FIELD_OP(Name, Kernel_Impl)                                       \
    template<typename T>                                                         \
    struct Name##Op                                                              \
    {                                                                            \
        template<typename Executor>                                              \
        void                                                                     \
        operator()(const Executor& exec, Field<T>& a, const Field<T>& b)         \
        {                                                                        \
            using executor = typename Executor::exec;                            \
            auto a_f = a.field();                                                \
            const auto b_f = b.field();                                          \
            Kokkos::parallel_for(                                                \
                Kokkos::RangePolicy<executor>(0, a_f.size()),                    \
                KOKKOS_CLASS_LAMBDA(const int i) { Kernel_Impl; }                \
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
                Kernel_Impl;                                                     \
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

BINARY_FIELD_OP(add, a_f[i] += b_f[i]);
BINARY_FIELD_OP(sub, a_f[i] -= b_f[i]);
BINARY_FIELD_OP(mul, a_f[i] *= b_f[i]);

#undef FIELD_FIELD_OP


} // namespace NeoFOAM
