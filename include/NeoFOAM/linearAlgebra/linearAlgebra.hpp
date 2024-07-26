// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoFOAM/fields/field.hpp"


namespace NeoFOAM
{

struct Dim
{
    size_t numRows;
    size_t numCols;
};


template<typename IndexType>
struct MatrixCoordinate
{
    IndexType row;
    IndexType col;
};

template<typename ValueType, typename IndexType>
struct MatrixEntry
{
    IndexType row;
    IndexType col;
    ValueType value;
};

std::shared_ptr<gko::Executor> getGkoExecutor(Executor exec)
{
    return std::visit(
        [](auto concreteExec)
        { return gko::ext::kokkos::create_executor(concreteExec.underlyingExec()); },
        exec
    );
}


template<typename ValueType, typename IndexType>
struct MatrixAssemblyInterface
{
    MatrixAssemblyInterface(Executor exec, size_t numRows, size_t numCols, size_t nnzEstimation);

    KOKKOS_FUNCTION void insert(size_t nnzId, MatrixEntry<ValueType, IndexType> entry) const;
};

template<typename ValueType, typename IndexType>
struct MatrixInterface
{
    MatrixInterface(MatrixAssemblyInterface<ValueType, IndexType>&&);

    void apply(const Field<ValueType>&, Field<ValueType>&);
};


template<typename ValueType, typename IndexType>
struct GinkgoMatrixAssembly
{
    GinkgoMatrixAssembly(Executor exec, size_t numRows, size_t numCols, size_t nnzEstimation)
        : data(getGkoExecutor(exec), gko::dim<2>(numRows, numCols), nnzEstimation)
    {
        auto size = data.get_size();
        auto arrays = data.empty_out();
        arrays.row_idxs.fill(0);
        arrays.col_idxs.fill(0);
        arrays.values.fill(0);
        data = gko::device_matrix_data<ValueType, IndexType>(
            data.get_executor(),
            size,
            std::move(arrays.row_idxs),
            std::move(arrays.col_idxs),
            std::move(arrays.values)
        );
        rowIdxs = data.get_row_idxs();
        colIdxs = data.get_col_idxs();
        values = data.get_values();
    }

    KOKKOS_FUNCTION void insert(size_t nnzId, MatrixEntry<ValueType, IndexType> entry) const
    {
        rowIdxs[nnzId] = entry.row;
        colIdxs[nnzId] = entry.col;
        values[nnzId] = entry.value;
    }

    Executor exec() const { return exec_; }

    auto&& getUnderlyingData() && { return std::move(data); }

    auto& getUnderlyingData() & { return data; }

    void finalize()
    {
        data.sort_row_major();
        data.sum_duplicates();
    }

private:

    Executor exec_;
    gko::device_matrix_data<ValueType, label> data;

    IndexType* rowIdxs;
    IndexType* colIdxs;
    ValueType* values;
};


template<typename ValueType>
class Matrix
{
public:

    Matrix() : mtx(gko::matrix::Coo<ValueType, int>::create()) {}
    Matrix(GinkgoMatrixAssembly<ValueType, int>&& matrixAssembly)
        : mtx(gko::matrix::Coo<ValueType, int>::create(getGkoExecutor(matrixAssembly.exec())))
    {
        matrixAssembly.finalize();
        auto gkoData = std::move(matrixAssembly).getUnderlyingData();

        mtx->read(std::move(gkoData));
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

    auto* getUnderlyingData() { return mtx.get(); }

private:

    std::shared_ptr<gko::matrix::Coo<ValueType, label>> mtx;
};


}
