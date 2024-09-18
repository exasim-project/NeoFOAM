// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#if NF_WITH_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/linearAlgebra/utilities.hpp"


namespace NeoFOAM::la::ginkgo
{

std::shared_ptr<gko::Executor> getGkoExecutor(Executor exec)
{
    return std::visit(
        [](auto concreteExec)
        { return gko::ext::kokkos::create_executor(concreteExec.underlyingExec()); },
        exec
    );
}

template<typename ValueType, typename IndexType>
class Matrix;

template<typename ValueType, typename IndexType = int32_t>
class MatrixBuilder
{
    friend class Matrix<ValueType, IndexType>;

public:

    struct SymbolicAssembly : CompatibleWithAnyExecutor
    {
        SymbolicAssembly(MatrixBuilder& mat)
            : rowIdxs(mat.data.get_row_idxs()), colIdxs(mat.data.get_col_idxs())
        {}

        KOKKOS_FUNCTION void insert(size_t nnzId, MatrixCoordinate<IndexType> coordinate) const
        {
            rowIdxs[nnzId] = coordinate.row;
            colIdxs[nnzId] = coordinate.col;
        }

    private:

        IndexType* rowIdxs;
        IndexType* colIdxs;
    };

    struct NumericAssembly : CompatibleWithAnyExecutor
    {
        NumericAssembly(MatrixBuilder& mat) : values(mat.data.get_values()) {}

        KOKKOS_FUNCTION void insert(size_t nnzId, ValueType value) const { values[nnzId] = value; }

    private:

        ValueType* values;
    };

    struct Assembly : CompatibleWithAnyExecutor
    {
        Assembly(MatrixBuilder& mat) : symAssembly(mat), numAssembly(mat) {}

        KOKKOS_FUNCTION void insert(size_t nnzId, MatrixEntry<ValueType, IndexType> entry) const
        {
            symAssembly.insert(nnzId, {entry.row, entry.col});
            numAssembly.insert(nnzId, entry.value);
        }

    private:

        SymbolicAssembly symAssembly;
        NumericAssembly numAssembly;
    };

    MatrixBuilder(Executor exec, Dim dim, size_t nnzEstimation)
        : exec_(exec),
          data(getGkoExecutor(exec), gko::dim<2>(dim.numRows, dim.numCols), nnzEstimation)
    {
        auto size = data.get_size();
        auto arrays = data.empty_out();
        arrays.row_idxs.fill(0);
        arrays.col_idxs.fill(0);
        arrays.values.fill(ValueType {});
        data = gko::device_matrix_data<ValueType, IndexType>(
            data.get_executor(),
            size,
            std::move(arrays.row_idxs),
            std::move(arrays.col_idxs),
            std::move(arrays.values)
        );
    }

    SymbolicAssembly startSymbolicAssembly() { return {*this}; }

    NumericAssembly startNumericAssembly(SymbolicAssembly&& assembly) { return {*this}; }

    Assembly startAssembly() { return {*this}; }

    Executor getExecutor() const { return exec_; }

private:

    Executor exec_;
    gko::device_matrix_data<ValueType, IndexType> data;
};


template<typename ValueType, typename IndexType = int32_t>
class Matrix
{
public:

    Matrix(MatrixBuilder<ValueType, IndexType>&& builder)
        : exec_(builder.exec_),
          mtx(gko::matrix::Coo<ValueType, IndexType>::create(builder.data.get_executor()))
    {
        builder.data.sum_duplicates();
        builder.data.remove_zeros();
        mtx->read(std::move(builder.data));
    }

    void apply(const Field<ValueType>& in, Field<ValueType>& out)
    {
        auto wrapField = [](auto& field)
        {
            auto data = const_cast<ValueType*>(field.data());
            auto gkoExec = getGkoExecutor(field.exec());
            return gko::matrix::Dense<ValueType>::create(
                gkoExec,
                gko::dim<2>(field.size(), 1),
                gko::make_array_view(gkoExec, field.size(), data),
                1
            );
        };
        mtx->apply(wrapField(in), wrapField(out));
    }

    Executor getExecutor() const { return exec_; }

    auto* getUnderlyingData() { return mtx.get(); }

private:

    Executor exec_;

    std::shared_ptr<gko::matrix::Coo<ValueType, IndexType>> mtx;
};


}

#endif
