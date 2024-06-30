// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>

#include <iostream>
#include <span>

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/operations/operationsMacros.hpp"
#include "NeoFOAM/fields/fieldTypeDefs.hpp"

namespace NeoFOAM
{
namespace detail
{

}

/**
 * @class Field
 * @brief A class to contain the data and executors for a field and define some
 * basic operations.
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
     * */
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
     * */
    Field(const Executor& exec, std::vector<ValueType> in)
        : size_(in.size()), data_(nullptr), exec_(exec)
    {
        Executor host_exec = CPUExecutor();
        void* ptr = nullptr;
        std::visit(
            [this, &ptr](const auto& exec) { ptr = exec.alloc(this->size_ * sizeof(ValueType)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);

        if (exec_ != host_exec)
        {
            Kokkos::View<ValueType*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryUnmanaged>
                gpuView(data_, size_);
            Kokkos::View<ValueType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostView(
                in.data(), size_
            );

            Kokkos::deep_copy(gpuView, hostView);
        }
        else
        {
            std::copy(in.begin(), in.end(), data_);
        }
    }


    Field(const Field<ValueType>& rhs) : size_(rhs.size_), data_(nullptr), exec_(rhs.exec_)
    {
        void* ptr = nullptr;
        auto size = rhs.size_;
        std::visit(
            [this, &ptr, size](const auto& exec) { ptr = exec.alloc(size * sizeof(ValueType)); },
            exec_
        );
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
     * @brief Copies the data to a new field on a specific executor
     * @returns A copy of the field on the host.
     */
    [[nodiscard]] Field<ValueType> copyToExecutor(Executor dstExec) const
    {
        if (dstExec == exec_) return Field<ValueType>(*this);
        Field<ValueType> result(dstExec, size_);

        auto srcPtr = data_;
        auto dstPtr = result.data();
        auto size = size_;

        auto deepCopyVisitor = [size, srcPtr, dstPtr](const auto& srcExec, const auto& dstExec)
        {
            auto createView = [size]<typename ExecType>(const ExecType& exec, ValueType* ptr)
            {
                using ExecutionSpace = std::conditional<
                    std::is_same_v<GPUExecutor, ExecType>,
                    Kokkos::DefaultExecutionSpace,
                    Kokkos::HostSpace>::type;
                return Kokkos::View<ValueType*, ExecutionSpace, Kokkos::MemoryUnmanaged>(ptr, size);
            };

            Kokkos::deep_copy(createView(dstExec, dstPtr), createView(srcExec, srcPtr));
        };

        std::visit(deepCopyVisitor, exec_, dstExec);

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


    /**
     * @brief access to underlying data
     * @param i The index of cell in the field
     * @warning returned value might not be on host
     */
    const ValueType at(const size_t i) const { return *(data_ + i); }

    /**
     * @brief Function call operator
     * @param i The index of cell in the field
     * @returns The value at the index i
     *
     * @warning This function is not implemented
     */
    KOKKOS_FUNCTION
    ValueType& operator()(const int i)
    {
        // TODO not implemented
        throw std::runtime_error("Not implemented");
    }

    /**
     * @brief Function call operator
     * @param i The index of cell in the field
     * @returns The value at the index i
     *
     * @warning This function is not implemented
     */
    KOKKOS_FUNCTION
    const ValueType& operator()(const int i) const
    {
        // TODO not implemented
        exit(1);
    }

    /**
     * @brief Assignment operator, Sets the field values to that of the parsed
     * field.
     * @param rhs The field to copy from.
     *
     * @warning This field will be sized to the size of the parsed field.
     */
    void operator=(const Field<ValueType>& rhs)
    {

        if (this->size() != rhs.size())
        {
            this->setSize(rhs.size());
        }
        setField(*this, rhs.span());
    }

    /**
     * @brief Assignment operator, Sets the field values to that of the value.
     * @param rhs The value to set the field to.
     */
    void operator=(const ValueType& rhs) { fill(*this, rhs); }

    // arithmetic operator

    /**
     * @brief Arithmetic add operator, addition of a second field.
     * @param rhs The field to add with this field.
     * @returns The result of the addition.
     */
    [[nodiscard]] Field<ValueType> operator+(const Field<ValueType>& rhs)
    {
        Field<ValueType> result(exec_, size_);
        result = *this;
        add(result, rhs);
        return result;
    }

    /**
     * @brief Arithmetic subtraction operator, subtraction by a second field.
     * @param rhs The field to subtract from this field.
     * @returns The result of the subtraction.
     */
    [[nodiscard]] Field<ValueType> operator-(const Field<ValueType>& rhs)
    {
        Field<ValueType> result(exec_, size_);
        result = *this;
        sub(result, rhs);
        return result;
    }

    /**
     * @brief Arithmetic subtraction operator, subtraction by a second field.
     * @param rhs The field to subtract from this field.
     * @returns The result of the subtraction.
     */
    [[nodiscard]] Field<ValueType> operator*(const Field<scalar>& rhs)
    {
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

    // setter

    /**
     * @brief Set the Size of the field.
     * @param size The new size to set the field to.
     */
    void setSize(const size_t size)
    {
        void* ptr = nullptr;
        std::visit(
            [this, &ptr, size](const auto& exec)
            { ptr = exec.realloc(data_, size * sizeof(ValueType)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);
        size_ = size;
    }

    // getter

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

private:


    size_t size_;         //!< Size of the field.
    ValueType* data_;     //!< Pointer to the field data.
    const Executor exec_; //!< Executor associated with the field. (CPU, GPU, openMP, etc.)
};


} // namespace NeoFOAM
