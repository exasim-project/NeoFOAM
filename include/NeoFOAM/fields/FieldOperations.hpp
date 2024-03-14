// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>

namespace NeoFOAM
{

/* A functor to implement the apply operation
**
*/
template<typename T>
struct ApplyOp
{

    /* TODO can Funct be const
     *  */
    template<typename Executor, typename Func>
    void operator()(const Executor& exec, Field<T>& f, Func function_)
    {
        using executor = typename Executor::exec;
        auto s_f = f.field();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<executor>(0, s_f.size()),
            KOKKOS_CLASS_LAMBDA(const int i) { s_f[i] = function_(i); }
        );
    }

    template<typename Executor, typename Func>
    void operator()(const CPUExecutor& exec, Field<T>& f, Func function_)
    {
        auto s_f = f.field();
        for (int i = 0; i < s_f.size(); i++)
        {
            s_f[i] = function_(i);
        }
    }
};

template<typename T, typename Func>
void map(Field<T>& f, Func func)
{
    ApplyOp<T> func_;
    std::visit([&](const auto& exec)
               { func_(exec, f, func); },
               f.exec());
}

template<typename T>
struct fillOp
{
    T fillValue_;

    template<typename Executor>
    void operator()(const Executor& exec, Field<T>& f)
    {
        using executor = typename Executor::exec;
        auto s_f = f.field();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<executor>(0, s_f.size()),
            KOKKOS_CLASS_LAMBDA(const int i) { s_f[i] = fillValue_; }
        );
    }

    template<typename Executor>
    void operator()(const CPUExecutor& exec, Field<T>& f)
    {
        auto s_f = f.field();
        for (int i = 0; i < s_f.size(); i++)
        {
            s_f[i] = fillValue_;
        }
    }
};

template<typename T>
void fill(Field<T>& f, const T value)
{
    fillOp<T> fill_ {value};
    std::visit([&](const auto& exec)
               { fill_(exec, f); },
               f.exec());
}

template<typename T>
struct setFieldOp
{
    const Field<T>& rhsField_;

    template<typename Executor>
    void operator()(const Executor& exec, Field<T>& f)
    {
        using executor = typename Executor::exec;
        auto s_f = f.field();
        auto s_rhsField_ = rhsField_.field();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<executor>(0, s_f.size()),
            KOKKOS_CLASS_LAMBDA(const int i) { s_f[i] = s_rhsField_[i]; }
        );
    }

    template<typename Executor>
    void operator()(const CPUExecutor& exec, Field<T>& f)
    {
        auto s_f = f.field();
        auto s_rhsField_ = rhsField_.field();
        for (int i = 0; i < s_f.size(); i++)
        {
            s_f[i] = s_rhsField_[i];
        }
    }
};

template<typename T>
void setField(Field<T>& f, const Field<T>& rhsField_)
{
    if (f.size() != rhsField_.size())
    {
        f.setSize(rhsField_.size());
    }
    setFieldOp<T> setField_ {rhsField_};
    std::visit([&](const auto& exec)
               { setField_(exec, f); },
               f.exec());
}


/**
 * @brief Functor to hold the plattform portable addition kernel
 */
template<typename T>
struct addOp
{
    template<typename Executor>
    void operator()(const Executor& exec, Field<T>& a, const Field<T>& b)
    {
        using executor = typename Executor::exec;
        auto a_f = a.field();
        auto b_f = b.field();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<executor>(0, a_f.size()),
            KOKKOS_CLASS_LAMBDA(const int i) { a_f[i] = a_f[i] + b_f[i]; }
        );
    }

    // NOTE DONT MERGE do we need the CPU version of this here, should we call this a reference
    // implementation?
    template<typename Executor>
    void operator()(const CPUExecutor& exec, Field<T>& a, const Field<T>& b)
    {
        auto a_f = f.field();
        const auto b_f = b.field();
        for (int i = 0; i < a_f.size(); i++)
        {
            a_f[i] += b_f[i];
        }
    }
};

/**
 * @brief Arithmetic add function, performs addition of two fields
 * @param a In-out parameter,
 * @param b The second field
 */
template<typename T>
void add(Field<T>& a, const Field<T>& b)
{
    addOp<T> addField_;
    std::visit([&](const auto& exec)
               { addField_(exec, a, b); },
               a.exec());
}

template<typename T>
struct subOp
{
    template<typename Executor>
    void operator()(const Executor& exec, Field<T>& f, const Field<T>& rhsField)
    {
        using executor = typename Executor::exec;
        auto s_f = f.field();
        auto s_rhsField = rhsField.field();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<executor>(0, s_f.size()),
            KOKKOS_CLASS_LAMBDA(const int i) { s_f[i] = s_f[i] - s_rhsField[i]; }
        );
    }

    template<typename Executor>
    void operator()(const CPUExecutor& exec, Field<T>& f, const Field<T>& rhsField)
    {
        auto s_f = f.field();
        auto s_rhsField = rhsField.field();
        for (int i = 0; i < s_f.size(); i++)
        {
            s_f[i] = s_f[i] - s_rhsField[i];
        }
    }
};

template<typename T>
void sub(Field<T>& f, const Field<T>& rhsField)
{
    subOp<T> subField_;
    std::visit([&](const auto& exec)
               { subField_(exec, f, rhsField); },
               f.exec());
}

template<typename T>
struct mulOp
{
    // const Field<T>& rhsField_;

    template<typename Executor>
    void operator()(const Executor& exec, Field<T>& f, const Field<scalar>& rhsField)
    {
        using executor = typename Executor::exec;
        auto s_f = f.field();
        auto s_rhsField = rhsField.field();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<executor>(0, s_f.size()),
            KOKKOS_CLASS_LAMBDA(const int i) { s_f[i] = s_f[i] * s_rhsField[i]; }
        );
    }

    template<typename Executor>
    void operator()(const CPUExecutor& exec, Field<T>& f, const Field<scalar>& rhsField)
    {
        auto s_f = f.field();
        auto s_rhsField = rhsField.field();
        for (int i = 0; i < s_f.size(); i++)
        {
            s_f[i] = s_f[i] * s_rhsField[i];
        }
    }

    template<typename Executor>
    void operator()(const Executor& exec, Field<T>& f, scalar s)
    {
        using executor = typename Executor::exec;
        auto s_f = f.field();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<executor>(0, s_f.size()),
            KOKKOS_CLASS_LAMBDA(const int i) { s_f[i] = s_f[i] * s; }
        );
    }

    template<typename Executor>
    void operator()(const CPUExecutor& exec, Field<T>& f, scalar s)
    {
        auto s_f = f.field();
        for (int i = 0; i < s_f.size(); i++)
        {
            s_f[i] = s_f[i] * s;
        }
    }
};

template<typename T>
void mul(Field<T>& f, const Field<T>& rhsField)
{
    mulOp<T> mulField_;
    std::visit([&](const auto& exec)
               { mulField_(exec, f, rhsField); },
               f.exec());
}

template<typename T>
void mul(Field<T>& f, T s)
{
    mulOp<T> mulField_;
    std::visit([&](const auto& exec)
               { mulField_(exec, f, s); },
               f.exec());
}

} // namespace NeoFOAM
