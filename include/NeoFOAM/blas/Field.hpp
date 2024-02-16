// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "primitives/scalar.hpp"

#include "NeoFOAM/blas/executor/executor.hpp"
#include "NeoFOAM/blas/span.hpp"

namespace NeoFOAM
{
    template <typename T>
    class Field
    {
    public:

        size_t size() const
        {
            return size_;
        }

        Field(size_t size, const executor& exec)
        : size_(size)
        , exec_(exec)
        , data_(nullptr)
        {
            void* ptr = nullptr;
            std::visit([this,&ptr,size](const auto& exec) {
                ptr = exec.alloc(size * sizeof(T));
            }, exec_);
            data_ = static_cast<T*>(ptr);
        };

        ~Field()
        {
            std::visit([this](const auto& exec) {
                exec.free(data_);
            }, exec_);
        };

        void setSize(size_t size)
        {
            void* ptr = nullptr;
            std::visit([this,&ptr,size](const auto& exec) {
                ptr = exec.realloc(data_, size * sizeof(T));
            }, exec_);
            data_ = static_cast<T*>(ptr);
            size_ = size;
        }
      
        span<T> field()
        {
            return span<T>(data_, size_);
        }

        const span<T> field() const
        {
            return span<T>(data_, size_);
        }

        void operator=(const Field<T> &rhs)
        {
            // set the field from the rhs field and resize if necessary
            setField(*this,rhs);
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

        Field<T> operator+(const Field<T> &rhs)
        {
            Field<T> result(size_, exec_);
            result = *this;
            add(result,rhs);
            return result;
        }

        Field<T> operator-(const Field<T> &rhs)
        {
            Field<T> result(size_, exec_);
            result = *this;
            sub(result,rhs);
            return result;
        }

        Field<T> operator*(const Field<scalar> &rhs)
        {
            Field<T> result(size_, exec_);
            result = *this;
            mul(result,rhs);
            return result;
        }

        Field<T> operator*(const scalar rhs)
        {
            Field<T> result(size_, exec_);
            result = *this;
            mul(result,rhs);
            return result;
        }

        template <typename func>
        void apply(func f)
        {
            map(*this, f);
        }
        
        
        Field<T> copyToHost()
        {
            Field<T> result(size_, CPUExecutor{});
            if (!std::holds_alternative<GPUExecutor>(exec_))
            {
                result = *this;
            }
            else
            {
                Kokkos::View<T *, Kokkos::Cuda, Kokkos::MemoryUnmanaged> GPU_view(data_,size_);
                Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> result_view(result.data(),size_);
                Kokkos::deep_copy(result_view, GPU_view);
            }
            return result;
        }

        T* data()
        {
            return data_;
        }

        const executor& exec()
        {
            return exec_;
        }

        private:
            size_t size_;
            const executor exec_;
            T* data_;
    };


} // namespace NeoFOAM