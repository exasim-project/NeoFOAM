// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "primitives/scalar.hpp"

namespace NeoFOAM
{
    template <typename T>
    class deviceField
    {
    public:
        KOKKOS_FUNCTION
        deviceField(const deviceField<T> &rhs)
            : size_(rhs.size_), field_(rhs.field_)
        {
        }

        KOKKOS_FUNCTION
        deviceField(const Kokkos::View<T *> &field)
            : size_(field.size()), field_(field)
        {

        }

        deviceField(const std::string &name, const int size)
            : size_(size), field_(Kokkos::View<T *>(name, size))
        {

        }

        KOKKOS_FUNCTION
        T &operator()(const int i) const
        {
            return field_(i);
        }

        void operator=(const deviceField<T> &rhs)
        {
            Kokkos::parallel_for(
                size_, KOKKOS_CLASS_LAMBDA(const int i) {
                    field_(i) = rhs(i);
                });
            size_ = rhs.size_;
        }

        // move assignment operator
        deviceField<T> &operator=(deviceField<T> &&rhs)
        {
            if (this != &rhs)
            {
                field_ = std::move(rhs.field_);
                size_ = rhs.size_;
            }
            return *this;
        }

        deviceField<T> operator+(const deviceField<T> &rhs)
        {
            deviceField<T> result("result", size_);
            Kokkos::parallel_for(
                size_, KOKKOS_CLASS_LAMBDA(const int i) {
                    result(i) = field_(i) + rhs(i);
                });
            return result;
        }

        deviceField<T> operator-(const deviceField<T> &rhs)
        {
            deviceField<T> result("result", size_);
            Kokkos::parallel_for(
                size_, KOKKOS_CLASS_LAMBDA(const int i) {
                    result(i) = field_(i) - rhs(i);
                });
            return result;
        }

        deviceField<T> operator*(const deviceField<scalar> &rhs)
        {
            deviceField<T> result("result", size_);
            Kokkos::parallel_for(
                size_, KOKKOS_CLASS_LAMBDA(const int i) {
                    result(i) = field_(i) * rhs(i);
                });
            return result;
        }

        deviceField<T> operator*(const double rhs)
        {
            deviceField<T> result("result", size_);
            Kokkos::parallel_for(
                size_, KOKKOS_CLASS_LAMBDA(const int i) {
                    result(i) = field_(i) * rhs;
                });
            return result;
        }

        template <typename func>
        void apply(func f)
        {
            Kokkos::parallel_for(
                size_, KOKKOS_CLASS_LAMBDA(const int i) {
                    field_(i) = f(i);
                });
        }

        auto data()
        {
            return field_.data();
        }

        std::string name()
        {
            return field_.name();
        }

        auto field()
        {
            return field_;
        }
        int size()
        {
            return size_;
        }

    private:
        Kokkos::View<T *> field_;
        int size_;
    };
} // namespace NeoFOAM
