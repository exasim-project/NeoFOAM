// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>

namespace NeoFOAM
{

    template <typename T>
    struct fillOp
    {
        T fillValue_;

        template <typename Executor>
        void operator()(const Executor &exec, Field<T> &f)
        {
            using executor = typename Executor::exec;
            auto s_f = f.field();
            Kokkos::parallel_for(
                Kokkos::RangePolicy<executor>(0, s_f.size()), KOKKOS_CLASS_LAMBDA(const int i) {
                    s_f[i] = fillValue_;
                });
        }

        template <typename Executor>
        void operator()(const CPUExecutor& exec, Field<T>& f)
        {
            auto s_f = f.field();
            for (int i = 0; i < s_f.size(); i++)
            {
                s_f[i] = fillValue_;
            }
        }

    };

    template< typename T>
    void fill(Field< T >& f, T value)
    {
        fillOp<T> fill_(value);
        std::visit([&](const auto &exec) {
                fill_(exec, f);
            }, f.exec());
    }


    template <typename T>
    struct setFieldOp
    {
        const Field<T>& rhsField_;

        template <typename Executor>
        void operator()(const Executor &exec, Field<T> &f)
        {
            using executor = typename Executor::exec;
            auto s_f = f.field();
            auto s_rhsField_ = rhsField_.field();
            Kokkos::parallel_for(
                Kokkos::RangePolicy<executor>(0, s_f.size()), KOKKOS_CLASS_LAMBDA(const int i) {
                    s_f[i] = s_rhsField_[i];
                });
        }

        template <typename Executor>
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

    template< typename T>
    void setField(Field< T >& f, const Field< T >& rhsField_)
    {
        if (f.size() != rhsField_.size())
        {
            f.setSize(rhsField_.size());
        }
        setFieldOp<T> setField_(rhsField_);
        std::visit([&](const auto &exec) {
                setField_(exec, f);
            }, f.exec());
    }

    template <typename T>
    struct addOp
    {
        // const Field<T>& rhsField_;

        template <typename Executor>
        void operator()(const Executor &exec, Field<T> &f, const Field< T >& rhsField)
        {
            using executor = typename Executor::exec;
            auto s_f = f.field();
            auto s_rhsField = rhsField.field();
            Kokkos::parallel_for(
                Kokkos::RangePolicy<executor>(0, s_f.size()), KOKKOS_CLASS_LAMBDA(const int i) {
                    s_f[i] = s_f[i] + s_rhsField[i];
                });
        }

        template <typename Executor>
        void operator()(const CPUExecutor& exec, Field<T>& f, const Field< T >& rhsField)
        {
            auto s_f = f.field();
            auto s_rhsField = rhsField.field();
            for (int i = 0; i < s_f.size(); i++)
            {
                s_f[i] = s_f[i] + s_rhsField[i];
            }
        }

    };

    template< typename T>
    void add(Field< T >& f, const Field< T >& rhsField)
    {
        addOp<T> addField_;
        std::visit([&](const auto &exec) {
                addField_(exec, f, rhsField);
            }, f.exec());
    }

    template <typename T>
    struct subOp
    {
        // const Field<T>& rhsField_;

        template <typename Executor>
        void operator()(const Executor &exec, Field<T> &f, const Field< T >& rhsField)
        {
            using executor = typename Executor::exec;
            auto s_f = f.field();
            auto s_rhsField = rhsField.field();
            Kokkos::parallel_for(
                Kokkos::RangePolicy<executor>(0, s_f.size()), KOKKOS_CLASS_LAMBDA(const int i) {
                    s_f[i] = s_f[i] - s_rhsField[i];
                });
        }

        template <typename Executor>
        void operator()(const CPUExecutor& exec, Field<T>& f, const Field< T >& rhsField)
        {
            auto s_f = f.field();
            auto s_rhsField = rhsField.field();
            for (int i = 0; i < s_f.size(); i++)
            {
                s_f[i] = s_f[i] - s_rhsField[i];
            }
        }

    };

    template< typename T>
    void sub(Field< T >& f, const Field< T >& rhsField)
    {
        subOp<T> subField_;
        std::visit([&](const auto &exec) {
                subField_(exec, f, rhsField);
            }, f.exec());
    }

    template <typename T>
    struct mulOp
    {
        // const Field<T>& rhsField_;

        template <typename Executor>
        void operator()(const Executor &exec, Field<T> &f, const Field< scalar >& rhsField)
        {
            using executor = typename Executor::exec;
            auto s_f = f.field();
            auto s_rhsField = rhsField.field();
            Kokkos::parallel_for(
                Kokkos::RangePolicy<executor>(0, s_f.size()), KOKKOS_CLASS_LAMBDA(const int i) {
                    s_f[i] = s_f[i] * s_rhsField[i];
                });
        }

        template <typename Executor>
        void operator()(const CPUExecutor& exec, Field<T>& f, const Field< scalar >& rhsField)
        {
            auto s_f = f.field();
            auto s_rhsField = rhsField.field();
            for (int i = 0; i < s_f.size(); i++)
            {
                s_f[i] = s_f[i] * s_rhsField[i];
            }
        }

        template <typename Executor>
        void operator()(const Executor &exec, Field<T> &f, scalar s)
        {
            using executor = typename Executor::exec;
            auto s_f = f.field();
            Kokkos::parallel_for(
                Kokkos::RangePolicy<executor>(0, s_f.size()), KOKKOS_CLASS_LAMBDA(const int i) {
                    s_f[i] = s_f[i] * s;
                });
        }

        template <typename Executor>
        void operator()(const CPUExecutor& exec, Field<T>& f, scalar s)
        {
            auto s_f = f.field();
            for (int i = 0; i < s_f.size(); i++)
            {
                s_f[i] = s_f[i] * s;
            }
        }

    };

    template< typename T>
    void mul(Field< T >& f, const Field< T >& rhsField)
    {
        mulOp<T> mulField_;
        std::visit([&](const auto &exec) {
                mulField_(exec, f, rhsField);
            }, f.exec());
    }

    template< typename T>
    void mul(Field< T >& f, T s)
    {
        mulOp<T> mulField_;
        std::visit([&](const auto &exec) {
                mulField_(exec, f, s);
            }, f.exec());
    }

} // namespace NeoFOAM