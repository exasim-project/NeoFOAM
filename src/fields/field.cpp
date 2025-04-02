// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/fields/field.hpp"

namespace NeoFOAM
{

/**
 * @brief A helper function to simplify the common pattern of copying between and to executor.
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

template<typename ValueType>
Field<ValueType>::Field(const Executor& exec, size_t size)
    : size_(size), data_(nullptr), exec_(exec)
{
    void* ptr = nullptr;
    std::visit(
        [&ptr, size](const auto& concreteExec)
        { ptr = concreteExec.alloc(size * sizeof(ValueType)); },
        exec_
    );
    data_ = static_cast<ValueType*>(ptr);
}


template<typename ValueType>
Field<ValueType>::Field(const Executor& exec, const ValueType* in, size_t size, Executor hostExec)
    : size_(size), data_(nullptr), exec_(exec)
{
    void* ptr = nullptr;
    std::visit(
        [&ptr, size](const auto& concreteExec)
        { ptr = concreteExec.alloc(size * sizeof(ValueType)); },
        exec_
    );
    data_ = static_cast<ValueType*>(ptr);
    std::visit(deepCopyVisitor<ValueType>(size_, in, data_), hostExec, exec_);
}

template<typename ValueType>
Field<ValueType>::Field(const Executor& exec, size_t size, ValueType value)
    : size_(size), data_(nullptr), exec_(exec)
{
    void* ptr = nullptr;
    std::visit(
        [&ptr, size](const auto& execu) { ptr = execu.alloc(size * sizeof(ValueType)); }, exec_
    );
    data_ = static_cast<ValueType*>(ptr);
    NeoFOAM::fill(*this, value);
}

template<typename ValueType>
Field<ValueType> Field<ValueType>::copyToExecutor(Executor dstExec) const
{
    if (dstExec == exec_) return *this;

    Field<ValueType> result(dstExec, size_);
    std::visit(deepCopyVisitor(size_, data_, result.data()), exec_, dstExec);

    return result;
}

template<typename ValueType>
Field<ValueType> Field<ValueType>::copyToHost() const
{
    return copyToExecutor(SerialExecutor());
}

template<typename ValueType>
void Field<ValueType>::copyToHost(Field<ValueType>& result) const
{
    NF_DEBUG_ASSERT(result.size() == size_, "Parsed Field size not the same as current field size");
    result = copyToExecutor(SerialExecutor());
}


#define NF_DECLARE_FIELD(TYPENAME) template class Field<TYPENAME>


NF_FOR_ALL_INDEX_TYPES(NF_DECLARE_FIELD);
NF_FOR_ALL_VALUE_TYPES(NF_DECLARE_FIELD);

}
