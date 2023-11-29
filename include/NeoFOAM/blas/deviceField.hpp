// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>

namespace NeoFOAM
{
    template <typename T, int dims>
    class deviceField
    {
    public:
        deviceField(const deviceField<T, dims> &rhs)
            : name_(rhs.name_), size_(rhs.size_), field_(rhs.field_)
        {
        }

        deviceField(deviceField<T, dims> &&rhs)
            : name_(std::move(rhs.name_)), size_(rhs.size_), field_(std::move(rhs.field_))
        {
            // Reset rhs to default state
            rhs.size_ = 0;
            rhs.name_ = "";
        }

        deviceField(const std::string &name, const int size, const bool init = true)
            : name_(name), size_(size)
        {
            if (init)
            {
                field_ = Kokkos::View<T *[dims]>(name, size);
                Kokkos::parallel_for(
                    size, KOKKOS_LAMBDA(const int i) {
                        for (int j = 0; j < dims; ++j)
                        {
                            field_(i, j) = 0;
                        }
                    });
            }
            else
            {
                field_ = Kokkos::View<T *[dims]>(name, size);
            }
        }

        KOKKOS_FORCEINLINE_FUNCTION
        double &operator()(const int i) const
        {
            return field_(i, 0);
        }

        KOKKOS_FORCEINLINE_FUNCTION
        double &operator()(const int i, const int j) const
        {
            return field_(i, j);
        }

        void operator=(const deviceField<T, dims> &rhs)
        {
            Kokkos::parallel_for(
                size_, KOKKOS_LAMBDA(const int i) {
                    for (int j = 0; j < dims; ++j)
                    {
                        field_(i, j) = rhs(i, j);
                    }
                });
            size_ = rhs.size_;
            name_ = rhs.name_;
        }

        // move assignment operator
        deviceField<T, dims> &operator=(deviceField<T, dims> &&rhs)
        {
            if (this != &rhs)
            {
                field_ = std::move(rhs.field_);
                size_ = rhs.size_;
                name_ = rhs.name_;
            }
            return *this;
        }

        deviceField<T, dims> operator+(const deviceField<T, dims> &rhs)
        {
            deviceField<T, dims> result("result", size_, false);
            Kokkos::parallel_for(
                size_, KOKKOS_LAMBDA(const int i) {
                    for (int j = 0; j < dims; ++j)
                    {
                        result(i, j) = field_(i, j) + rhs(i, j);
                    }
                });
            return result;
        }

        deviceField<T, dims> operator-(const deviceField<T, dims> &rhs)
        {
            deviceField<T, dims> result("result", size_, false);
            Kokkos::parallel_for(
                size_, KOKKOS_LAMBDA(const int i) {
                    for (int j = 0; j < dims; ++j)
                    {
                        result(i, j) = field_(i, j) - rhs(i, j);
                    }
                });
            return result;
        }

        deviceField<T, dims> operator*(const deviceField<T, 1> &rhs)
        {
            deviceField<T, dims> result("result", size_, false);
            Kokkos::parallel_for(
                size_, KOKKOS_LAMBDA(const int i) {
                    for (int j = 0; j < dims; ++j)
                    {
                        result(i, j) = field_(i, j) * rhs(i);
                    }
                });
            return result;
        }

        deviceField<T, dims> operator*(const double rhs)
        {
            deviceField<T, dims> result("result", size_, false);
            Kokkos::parallel_for(
                size_, KOKKOS_LAMBDA(const int i) {
                    for (int j = 0; j < dims; ++j)
                    {
                        result(i, j) = field_(i, j) * rhs;
                    }
                });
            return result;
        }

        template <typename func>
        void apply(func f)
        {
            Kokkos::parallel_for(
                size_, KOKKOS_LAMBDA(const int i) {
                    for (int j = 0; j < dims; ++j)
                    {
                        field_(i, j) = f(i, j);
                    }
                });
        }

        auto data()
        {
            return field_.data();
        }
        Kokkos::View<T *[dims]> field_;
    private:
        
        std::string name_;
        int size_;
    };
} // namespace NeoFOAM
