// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "primitives/scalar.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

namespace NeoFOAM {

/* DeviceField is a Field class that handles data on devices.
 *
 */
template <typename T> class DeviceField {
public:
  // Constructor
  KOKKOS_FUNCTION
  DeviceField(const DeviceField<T> &rhs)
      : size_(rhs.size_), field_(rhs.field_) {}

  DeviceField(const std::string &name, const int size)
      : size_(size), field_(Kokkos::View<T *>(name, size)) {}

  KOKKOS_FUNCTION
  T &operator()(const int i) const { return field_(i); }

  KOKKOS_FUNCTION
  const T &operator()(const int i) const { return field_(i); }

  void operator=(const DeviceField<T> &rhs) {
    Kokkos::parallel_for(
        size_, KOKKOS_CLASS_LAMBDA(const int i) { field_(i) = rhs(i); });
    size_ = rhs.size_;
  }

  // move assignment operator
  DeviceField<T> &operator=(DeviceField<T> &&rhs) {
    if (this != &rhs) {
      field_ = std::move(rhs.field_);
      size_ = rhs.size_;
    }
    return *this;
  }

  // Arithmetic operations
  DeviceField<T> operator+(const DeviceField<T> &rhs) {
    DeviceField<T> result("result", size_);
    Kokkos::parallel_for(
        size_,
        KOKKOS_CLASS_LAMBDA(const int i) { result(i) = field_(i) + rhs(i); });
    return result;
  }

  DeviceField<T> operator-(const DeviceField<T> &rhs) {
    DeviceField<T> result("result", size_);
    Kokkos::parallel_for(
        size_,
        KOKKOS_CLASS_LAMBDA(const int i) { result(i) = field_(i) - rhs(i); });
    return result;
  }

  DeviceField<T> operator*(const DeviceField<scalar> &rhs) {
    DeviceField<T> result("result", size_);
    Kokkos::parallel_for(
        size_,
        KOKKOS_CLASS_LAMBDA(const int i) { result(i) = field_(i) * rhs(i); });
    return result;
  }

  DeviceField<T> operator*(const double rhs) {
    DeviceField<T> result("result", size_);
    Kokkos::parallel_for(
        size_,
        KOKKOS_CLASS_LAMBDA(const int i) { result(i) = field_(i) * rhs; });
    return result;
  }

  template <typename func> void apply(func f) {
    Kokkos::parallel_for(
        size_, KOKKOS_CLASS_LAMBDA(const int i) { field_(i) = f(i); });
  }

  // getter
  auto data() { return field_.data(); }

  std::string name() { return field_.name(); }

  auto field() { return field_; }

  int size() { return size_; }

private:
  Kokkos::View<T *> field_;
  int size_;
};
} // namespace NeoFOAM
