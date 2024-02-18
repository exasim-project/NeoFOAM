// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "primitives/scalar.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

#include "NeoFOAM/blas/executor/executor.hpp"
#include "NeoFOAM/blas/span.hpp"

namespace NeoFOAM {

/* Field is class that handles field data. Via the executor the location of
 * allocation can be selected.
 *
 */
template <typename T> class Field {
public:
  /* Create a Field with a given size on an executor
   *
   * @param exec  Executor associated to the matrix
   * @param size  size of the matrix
   * */
  Field(const executor &exec, size_t size)
      : size_(size), exec_(exec), data_(nullptr) {
    void *ptr = nullptr;
    std::visit([this, &ptr,
                size](const auto &exec) { ptr = exec.alloc(size * sizeof(T)); },
               exec_);
    data_ = static_cast<T *>(ptr);
  };

  ~Field() {
    std::visit([this](const auto &exec) { exec.free(data_); }, exec_);
  };

  /* Performs a piece transformation of the field data by a given function
  **
  ** @param f, The function to map over the field. Ideally a KOKKOS_LAMBA
  */
  template <typename func> void apply(func f) { map(*this, f); }

  /* Forces a copy back to the host
   *
   * @returns a field on the host with the copied data
   * */
  [[nodiscard]] Field<T> copyToHost() {
    Field<T> result(size_, CPUExecutor{});
    return this->copyToHost(result);
  }

  /* Forces a copy back to the host
   *
   * @param a field to store the copied data
   * @returns a field on the host with the copied data
   * */
  [[nodiscard]] void copyToHost(Field<T> result) {
    if (!std::holds_alternative<GPUExecutor>(exec_)) {
      // TODO how does that work? It also not clear to me if the outparam or the
      // return type should be used
      result = *this;
      // TODO should we have a
      //  return result;
      //  here for symmetry
    } else {
      Kokkos::View<T *, Kokkos::Cuda, Kokkos::MemoryUnmanaged> GPU_view(data_,
                                                                        size_);
      Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> result_view(
          result.data(), size_);
      Kokkos::deep_copy(result_view, GPU_view);
    }
  }

  // // move assignment operator
  // Field<T> &operator=(Field<T> &&rhs)
  // {
  //     if (this != &rhs)
  //     {
  //         field_ = std::move(rhs.field_);
  //         size_ = rhs.size_;
  //     }
  //     return *this;
  // }

  /* Copies the content of the rhs field*/
  void operator=(const Field<T> &rhs) {
    // set the field from the rhs field and resize if necessary
    setField(*this, rhs);
  }

  /* Fills the field data with the given rhs value*/
  void operator=(const T &rhs) {
    // set the field from the rhs field and resize if necessary
    fill(*this, rhs);
  }

  // arithmetic operator

  /* A piecewise addition operation.
  **
  */
  [[nodiscard]] Field<T> operator+(const Field<T> &rhs) {
    Field<T> result(size_, exec_);
    result = *this;
    add(result, rhs);
    return result;
  }

  /* A piecewise subtraction operation.
  **
  */
  [[nodiscard]] Field<T> operator-(const Field<T> &rhs) {
    Field<T> result(size_, exec_);
    result = *this;
    sub(result, rhs);
    return result;
  }

  /* A piecewise multiplication operation.
  **
  */
  [[nodiscard]] Field<T> operator*(const Field<scalar> &rhs) {
    Field<T> result(size_, exec_);
    result = *this;
    mul(result, rhs);
    return result;
  }

  /* A scalar multiplication operation.
  **
  */
  [[nodiscard]] Field<T> operator*(const scalar rhs) {
    Field<T> result(size_, exec_);
    result = *this;
    mul(result, rhs);
    return result;
  }

  // setter

  void setSize(size_t size) {
    void *ptr = nullptr;
    std::visit(
        [this, &ptr, size](const auto &exec) {
          ptr = exec.realloc(data_, size * sizeof(T));
        },
        exec_);
    data_ = static_cast<T *>(ptr);
    size_ = size;
  }

  // getter
  T *data() { return data_; }

  const T *data() const { return data_; }

  const executor &exec() { return exec_; }

  size_t size() const { return size_; }

  span<T> field() { return span<T>(data_, size_); }

  const span<T> field() const { return span<T>(data_, size_); }

private:
  size_t size_;
  const executor exec_;
  T *data_;
};

} // namespace NeoFOAM
