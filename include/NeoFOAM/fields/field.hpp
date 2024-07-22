// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>

#include <iostream>
#include <span>

#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/types.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/operations/operationsMacros.hpp"
#include "NeoFOAM/fields/fieldTypeDefs.hpp"

namespace NeoFOAM
{

namespace detail
{
/**
 * @brief A helper function to simplify the common pattern of copying between to executor.
 * @param size The number of elements to copy.
 * @param srcPtr Pointer to the original block of memory.
 * @param dstPtr Pointer to the target block of memory.
 * @tparam T The type of the underlying elements.
 * @returns A function that takes a source and an destination executor
 */
template<StorageType T>
auto deepCopyVisitor(size_t size, const T* srcPtr, T* dstPtr)
{
    return [size, srcPtr, dstPtr](const auto& srcExec, const auto& dstExec)
    {
        Kokkos::deep_copy(
            dstExec.createKokkosView(dstPtr, size), srcExec.createKokkosView(srcPtr, size)
        );
    };
}
}

/**
 * @class Field
 * @brief A class to contain the data and executors for a field and define some basic operations.
 *
 * @ingroup Fields
 */
template<StorageType T>
class Field
{

public:

    /**
     * @brief Create a Field with a given size on an executor
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Field(const Executor& exec, size_t size) : size_(size), data_(nullptr), exec_(exec)
    {
        void* ptr = nullptr;
        std::visit(
            [this, &ptr, size](const auto& concreteExec)
            { ptr = concreteExec.alloc(size * sizeof(T)); },
            exec_
        );
        data_ = static_cast<T*>(ptr);
    }

    /**
     * @brief Create a Field with a given size on an executor
     * @param exec  Executor associated to the matrix
     * @param in a vector of elements to copy over
     */
    Field(const Executor& exec, std::vector<T> in) : size_(in.size()), data_(nullptr), exec_(exec)
    {
        Executor hostExec = CPUExecutor();
        void* ptr = nullptr;
        std::visit(
            [this, &ptr](const auto& concreteExec)
            { ptr = concreteExec.alloc(this->size_ * sizeof(T)); },
            exec_
        );
        data_ = static_cast<T*>(ptr);

        std::visit(detail::deepCopyVisitor(size_, in.data(), data_), hostExec, exec);
    }

    /**
     * @brief Copy constructor, creates a new field with the same size and data as the parsed field.
     * @param rhs The field to copy from.
     */
    Field(const Field<T>& rhs) : data_(nullptr), exec_(rhs.exec_)
    {
        resize(rhs.size_);
        setField(*this, rhs.span());
    }

    /**
     * @brief Destroy the Field object.
     */
    ~Field()
    {
        std::visit([this](const auto& exec) { exec.free(data_); }, exec_);
        data_ = nullptr;
    }

    /**
     * @brief applies a functor, transformation, to the field
     * @param f The functor to map over the field.
     * @note Ideally the f should be a KOKKOS_LAMBA
     */
    template<typename func>
    void apply(func f)
    {
        map(*this, f);
    }

    /**
     * @brief Copies the data to a new field on a specific executor.
     * @param dstExec The executor on which the data should be copied.
     * @returns A copy of the field on the host.
     */
    [[nodiscard]] Field<T> copyToExecutor(Executor dstExec) const
    {
        if (dstExec == exec_) return Field<T>(*this);

        Field<T> result(dstExec, size_);
        std::visit(detail::deepCopyVisitor(size_, data_, result.data()), exec_, dstExec);

        return result;
    }

    /**
     * @brief Returns a copy of the field back to the host.
     * @returns A copy of the field on the host.
     */
    [[nodiscard]] Field<T> copyToHost() const { return copyToExecutor(CPUExecutor()); }

    /**
     * @brief Copies the data (from anywhere) to a parsed host field.
     * @param result The field into which the data must be copied. Must be
     * sized.
     *
     * @warning exits if the size of the result field is not the same as the
     * source field.
     */
    void copyToHost(Field<T>& result)
    {
        NF_DEBUG_ASSERT(
            result.size() == size_, "Parsed Field size not the same as current field size"
        );
        result = copyToExecutor(CPUExecutor());
    }

    /**
     * @brief Subscript operator
     * @param i The index of cell in the field
     * @returns The value at the index i
     */
    KOKKOS_FUNCTION
    T& operator[](const size_t i) { return data_[i]; }

    /**
     * @brief Subscript operator
     * @param i The index of cell in the field
     * @returns The value at the index i
     */
    KOKKOS_FUNCTION
    const T& operator[](const size_t i) const { return data_[i]; }

    /**
     * @brief Function call operator
     * @param i The index of cell in the field
     * @returns The value at the index i
     */
    KOKKOS_FUNCTION
    T& operator()(const size_t i) { return data_[i]; }

    /**
     * @brief Function call operator
     * @param i The index of cell in the field
     * @returns The value at the index i
     */
    KOKKOS_FUNCTION
    const T& operator()(const size_t i) const { return data_[i]; }

    /**
     * @brief Assignment operator, Sets the field values to that of the passed value.
     * @param rhs The value to set the field to.
     */
    void operator=(const T& rhs) { fill(*this, rhs); }

    /**
     * @brief Assignment operator, Sets the field values to that of the parsed field.
     * @param rhs The field to copy from.
     *
     * @warning This field will be sized to the size of the parsed field.
     */
    void operator=(const Field<T>& rhs)
    {
        NF_ASSERT(exec_ == rhs.exec_, "Executors are not the same");
        if (this->size() != rhs.size())
        {
            this->resize(rhs.size());
        }
        setField(*this, rhs.span());
    }

    /**
     * @brief Arithmetic add operator, addition of a second field.
     * @param rhs The field to add with this field.
     * @returns The result of the addition.
     */
    Field<T>& operator+=(const Field<T>& rhs)
    {
        validateOtherField(rhs);
        add(*this, rhs);
        return *this;
    }

    /**
     * @brief Arithmetic subtraction operator, subtraction by a second field.
     * @param rhs The field to subtract from this field.
     * @returns The result of the subtraction.
     */
    Field<T>& operator-=(const Field<T>& rhs)
    {
        validateOtherField(rhs);
        sub(*this, rhs);
        return *this;
    }

    /**
     * @brief Arithmetic multiply operator, multiply by a second field.
     * @param rhs The field to subtract from this field.
     * @returns The result of the multiply.
     */
    [[nodiscard]] Field<T> operator*(const Field<scalar>& rhs)
    {
        validateOtherField(rhs);
        Field<T> result(exec_, size_);
        result = *this;
        mul(result, rhs);
        return result;
    }

    /**
     * @brief Arithmetic multiply operator, multiplies every cell in the field
     * by a scalar.
     * @param rhs The scalar to multiply with the field.
     * @returns The result of the multiplication.
     */
    [[nodiscard]] Field<T> operator*(const scalar rhs)
    {
        Field<T> result(exec_, size_);
        result = *this;
        scalarMul(result, rhs);
        return result;
    }

    /**
     * @brief Resizes the field to a new size.
     * @param size The new size to set the field to.
     */
    void resize(const size_t size)
    {
        void* ptr = nullptr;
        if (!empty())
        {
            std::visit(
                [this, &ptr, size](const auto& exec)
                { ptr = exec.realloc(data_, size * sizeof(T)); },
                exec_
            );
        }
        else
        {
            std::visit(
                [this, &ptr, size](const auto& exec) { ptr = exec.alloc(size * sizeof(T)); }, exec_
            );
        }
        data_ = static_cast<T*>(ptr);
        size_ = size;
    }

    /**
     * @brief Direct access to the underlying field data
     * @return Pointer to the first cell data in the field.
     */
    [[nodiscard]] T* data() { return data_; }

    /**
     * @brief Direct access to the underlying field data
     * @return Pointer to the first cell data in the field.
     */
    [[nodiscard]] const T* data() const { return data_; }

    /**
     * @brief Gets the executor associated with the field.
     * @return Reference to the executor.
     */
    [[nodiscard]] const Executor& exec() const { return exec_; }

    /**
     * @brief Gets the size of the field.
     * @return The size of the field.
     */
    [[nodiscard]] size_t size() const { return size_; }

    /**
     * @brief Checks if the field is empty.
     * @return True if the field is empty, false otherwise.
     */
    [[nodiscard]] bool empty() const { return size() == 0; }

    /**
     * @brief Gets the field as a span.
     * @return Span of the field.
     */
    [[nodiscard]] std::span<T> span() { return std::span<T>(data_, size_); }

    /**
     * @brief Gets the field as a span.
     * @return Span of the field.
     */
    [[nodiscard]] std::span<const T> span() const { return std::span<const T>(data_, size_); }

    /**
     * @brief Gets a sub view of the field as a span.
     * @return Span of the field.
     */
    [[nodiscard]] std::span<T> span(std::pair<size_t, size_t> range)
    {
        return std::span<T>(data_ + range.first, range.second - range.first);
    }

    /**
     * @brief Gets a sub view of the field as a span.
     * @return Span of the field.
     */
    [[nodiscard]] std::span<const T> span(std::pair<size_t, size_t> range) const
    {
        return std::span<const T>(data_ + range.first, range.second - range.first);
    }

    /**
     * @brief Gets the range of the field.
     * @return The range of the field {0, size()}.
     */
    [[nodiscard]] std::pair<size_t, size_t> range() const { return {0, size()}; }

private:

    size_t size_ {0};     //!< Size of the field.
    T* data_ {nullptr};   //!< Pointer to the field data.
    const Executor exec_; //!< Executor associated with the field. (CPU, GPU, openMP, etc.)

    /**
     * @brief Checks if two fields are the same size and have the same executor.
     * @param rhs The field to compare with.
     */
    void validateOtherField(const Field<T>& rhs) const
    {
        NF_DEBUG_ASSERT(size() == rhs.size(), "Fields are not the same size.");
        NF_DEBUG_ASSERT(exec() == rhs.exec(), "Executors are not the same.");
    }
};

/**
 * @brief Arithmetic add operator, addition of two fields.
 * @param lhs The field to add with this field.
 * @param rhs The field to add with this field.
 * @returns The result of the addition.
 */
template<StorageType T>
[[nodiscard]] Field<T> operator+(Field<T> lhs, const Field<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

/**
 * @brief Arithmetic subtraction operator, subtraction one field from another.
 * @param lhs The field to subtract from.
 * @param rhs The field to subtract by.
 * @returns The result of the subtraction.
 */
template<StorageType T>
[[nodiscard]] Field<T> operator-(Field<T> lhs, const Field<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

} // namespace NeoFOAM
