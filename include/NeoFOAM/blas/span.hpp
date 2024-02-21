// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

namespace NeoFOAM {
template <typename T> class span {
public:
  KOKKOS_FUNCTION
  span(T *data, size_t size) : data_(data), size_(size) {}

  KOKKOS_INLINE_FUNCTION
  T &operator[](size_t i) const { return data_[i]; }

  KOKKOS_FUNCTION
  T *data() const { return data_; }

  KOKKOS_FUNCTION
  size_t size() const { return size_; }

private:
  T *data_;
  size_t size_;
};

} // namespace NeoFOAM