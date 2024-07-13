// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>

#include <iostream>
#include <span>

#include "NeoFOAM/core/error.hpp"
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
 * @tparam ValueType The type of the underlying elements.
 * @returns A function that takes a source and an destination executor
 */
template<typename ValueType>
auto deepCopyVisitor(size_t size, const ValueType* srcPtr, ValueType* dstPtr)
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
template<typename ValueType>
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
            [this, &ptr, size](const auto& exec) { ptr = exec.alloc(size * sizeof(ValueType)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);
    }

    /**
     * @brief Create a Field with a given size on an executor
     * @param exec  Executor associated to the matrix
     * @param in a vector of elements to copy over
     */
    Field(const Executor& exec, std::vector<ValueType> in)
        : size_(in.size()), data_(nullptr), exec_(exec)
    {
        Executor hostExec = CPUExecutor();
        void* ptr = nullptr;
        std::visit(
            [this, &ptr](const auto& exec) { ptr = exec.alloc(this->size_ * sizeof(ValueType)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);

        std::visit(detail::deepCopyVisitor(size_, in.data(), data_), hostExec, exec);
    }

    /**
     * @brief Copy constructor, creates a new field with the same size and data as the parsed field.
     * @param rhs The field to copy from.
     */
    Field(const Field<ValueType>& rhs) : size_(rhs.size_), data_(nullptr), exec_(rhs.exec_)
    {
        NF_ASSERT(exec_ == rhs.exec_, "Executors are not the same");
        void* ptr = nullptr;
        auto size = rhs.size_;
        std::visit(
            [this, &ptr, size](const auto& exec) { ptr = exec.alloc(size * sizeof(ValueType)); },
            exec_
        );
        setSize(rhs.size_); // CHECK THIS with above
        data_ = static_cast<ValueType*>(ptr);
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
    [[nodiscard]] Field<ValueType> copyToExecutor(Executor dstExec) const
    {
        if (dstExec == exec_) return Field<ValueType>(*this);

        Field<ValueType> result(dstExec, size_);
        std::visit(detail::deepCopyVisitor(size_, data_, result.data()), exec_, dstExec);

        return result;
    }

    /**
     * @brief Returns a copy of the field back to the host.
     * @returns A copy of the field on the host.
     */
    [[nodiscard]] Field<ValueType> copyToHost() const { return copyToExecutor(CPUExecutor()); }

    /**
     * @brief Copies the data (from anywhere) to a parsed host field.
     * @param result The field into which the data must be copied. Must be
     * sized.
     *
     * @warning exits if the size of the result field is not the same as the
     * source field.
     */
    void copyToHost(Field<ValueType>& result)
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
    ValueType& operator[](const int i) { return data_[i]; }

    /**
     * @brief Subscript operator
     * @param i The index of cell in the field
     * @returns The value at the index i
     */
    KOKKOS_FUNCTION
    const ValueType& operator[](const int i) const { return data_[i]; }

    /**
     * @brief Function call operator
     * @param i The index of cell in the field
     * @returns The value at the index i
     */
    KOKKOS_FUNCTION
    ValueType& operator()(const int i) { return data_[i]; }

    /**
     * @brief Function call operator
     * @param i The index of cell in the field
     * @returns The value at the index i
     */
    KOKKOS_FUNCTION
    const ValueType& operator()(const int i) const { return data_[i]; }

    /**
     * @brief Assignment operator, Sets the field values to that of the passed value.
     * @param rhs The value to set the field to.
     */
    void operator=(const ValueType& rhs) { fill(*this, rhs); }

    /**
     * @brief Assignment operator, Sets the field values to that of the parsed field.
     * @param rhs The field to copy from.
     *
     * @warning This field will be sized to the size of the parsed field.
     */
    void operator=(const Field<ValueType>& rhs)
    {
        NF_ASSERT(exec_ == rhs.exec_, "Executors are not the same");
        if (this->size() != rhs.size())
        {
            this->setSize(rhs.size());
        }
        setField(*this, rhs.span());
    }

    /**
     * @brief Arithmetic add operator, addition of a second field.
     * @param rhs The field to add with this field.
     * @returns The result of the addition.
     */
    Field<ValueType>& operator+=(const Field<ValueType>& rhs)
    {
        NF_DEBUG_ASSERT(size() == rhs.size(), "Fields are not the same size.");
        NF_DEBUG_ASSERT(exec_ == rhs.exec(), "Executors are not the same.");
        add(*this, rhs);
        return *this;
    }

    /**
     * @brief Arithmetic subtraction operator, subtraction by a second field.
     * @param rhs The field to subtract from this field.
     * @returns The result of the subtraction.
     */
    Field<ValueType>& operator-=(const Field<ValueType>& rhs)
    {
        NF_DEBUG_ASSERT(size() == rhs.size(), "Fields are not the same size.");
        NF_DEBUG_ASSERT(exec_ == rhs.exec(), "Executors are not the same.");
        sub(*this, rhs);
        return *this;
    }

    /**
     * @brief Arithmetic multiply operator, multiply by a second field.
     * @param rhs The field to subtract from this field.
     * @returns The result of the multiply.
     */
    [[nodiscard]] Field<ValueType> operator*(const Field<scalar>& rhs)
    {
        NF_DEBUG_ASSERT(size() == rhs.size(), "Fields are not the same size.");
        NF_DEBUG_ASSERT(exec_ == rhs.exec(), "Executors are not the same.");
        Field<ValueType> result(exec_, size_);
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
    [[nodiscard]] Field<ValueType> operator*(const scalar rhs)
    {
        Field<ValueType> result(exec_, size_);
        result = *this;
        scalar_mul(result, rhs);
        return result;
    }

    /**
     * @brief Set the Size of the field.
     * @param size The new size to set the field to.
     */
    void setSize(const size_t size)
    {
        void* ptr = nullptr;
        if (!empty())
        {
            std::visit(
                [this, &ptr, size](const auto& exec)
                { ptr = exec.realloc(data_, size * sizeof(ValueType)); },
                exec_
            );
        }
        else
        {
            std::visit(
                [this, &ptr, size](const auto& exec)
                { ptr = exec.alloc(size * sizeof(ValueType)); },
                exec_
            );
        }
        data_ = static_cast<ValueType*>(ptr);
        size_ = size;
    }

    /**
     * @brief Direct access to the underlying field data
     * @return Pointer to the first cell data in the field.
     */
    [[nodiscard]] ValueType* data() { return data_; }

    /**
     * @brief Direct access to the underlying field data
     * @return Pointer to the first cell data in the field.
     */
    [[nodiscard]] const ValueType* data() const { return data_; }

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
    [[nodiscard]] std::span<ValueType> span() { return std::span<ValueType>(data_, size_); }
    /**
     * @brief Gets the field as a span.
     * @return Span of the field.
     */
    [[nodiscard]] const std::span<ValueType> span() const
    {
        return std::span<ValueType>(data_, size_);
    }

    /**
     * @brief Gets a sub view of the field as a span.
     * @return Span of the field.
     */
    [[nodiscard]] std::span<ValueType> span(std::pair<size_t, size_t> range)
    {
        return std::span<ValueType>(data_ + range.first, range.second - range.first);
    }

    /**
     * @brief Gets a sub view of the field as a span.
     * @return Span of the field.
     */
    [[nodiscard]] const std::span<ValueType> span(std::pair<size_t, size_t> range) const
    {
        return span(range);
    }

    /**
     * @brief Gets the range of the field.
     * @return The range of the field {0, size()}.
     */
    [[nodiscard]] inline std::pair<size_t, size_t> field::range() const { return {0, size()}; }

private:

    size_t size_ {0};           //!< Size of the field.
    ValueType* data_ {nullptr}; //!< Pointer to the field data.
    const Executor exec_;       //!< Executor associated with the field. (CPU, GPU, openMP, etc.)
};

/**
 * @brief Arithmetic add operator, addition of two fields.
 * @param lhs The field to add with this field.
 * @param rhs The field to add with this field.
 * @returns The result of the addition.
 */
template<typename T>
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
template<typename T>
[[nodiscard]] Field<T> operator-(Field<T> lhs, const Field<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

} // namespace NeoFOAM
