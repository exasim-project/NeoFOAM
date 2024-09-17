#pragma once

#include <petsc.h>
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/linearAlgebra/ginkgo.hpp"

namespace NeoFOAM::la::petsc
{


class Matrix;


class MatrixBuilder
{
    friend class Matrix;

public:

    struct SymbolicAssembly
    {
        SymbolicAssembly(MatrixBuilder& mat)
            : rowIdxs(mat.rowIdxs.data()), colIdxs(mat.colIdxs.data())
        {}

        KOKKOS_FUNCTION void insert(size_t nnzId, MatrixCoordinate<PetscInt> coordinate) const
        {
            rowIdxs[nnzId] = coordinate.row;
            colIdxs[nnzId] = coordinate.col;
        }

        static Executor getCompatibleExecutor(const Executor& exec)
        {
            if (std::holds_alternative<GPUExecutor>(exec))
            {
                return CPUExecutor {};
            }

            return exec;
        }

    private:

        PetscInt* rowIdxs;
        PetscInt* colIdxs;
    };

    struct NumericAssembly : CompatibleWithAnyExecutor
    {
        NumericAssembly(MatrixBuilder& mat) : values(mat.values.data()) {}

        KOKKOS_FUNCTION void insert(size_t nnzId, PetscScalar value) const
        {
            values[nnzId] = value;
        }

    private:

        PetscScalar* values;
    };

    SymbolicAssembly startSymbolicAssembly()
    {
        rowIdxs.resize(nnzEstimation);
        colIdxs.resize(nnzEstimation);
        values.resize(nnzEstimation);
        fill(rowIdxs, 0);
        fill(colIdxs, 0);
        fill(values, 0);
        return {*this};
    }

    NumericAssembly startNumericAssembly(SymbolicAssembly&& assembly)
    {
        MatSetPreallocationCOO(mat, rowIdxs.size(), rowIdxs.data(), colIdxs.data());
        rowIdxs.resize(0);
        colIdxs.resize(0);
        return {*this};
    }

    MatrixBuilder(Dim dim, size_t nnzEstimation)
        : rowIdxs(CPUExecutor {}, nnzEstimation), colIdxs(CPUExecutor {}, nnzEstimation),
          values(get_executor(), nnzEstimation)
    {
        MatCreate(MPI_COMM_WORLD, &mat);
        MatSetType(mat, MATSEQAIJKOKKOS);
        MatSetSizes(mat, dim.numRows, dim.numCols, dim.numRows, dim.numCols);
    }

    Executor get_executor() const { return GPUExecutor {}; }

private:

    Dim dim;
    size_t nnzEstimation;

    Field<PetscInt> rowIdxs;
    Field<PetscInt> colIdxs;
    Field<PetscScalar> values;

    Mat mat;
};


class Matrix
{
public:

    Matrix(MatrixBuilder&& builder) : mat(builder.mat)
    {
        builder.mat = nullptr;
        MatSetValuesCOO(mat, builder.values.data(), INSERT_VALUES);
        builder.rowIdxs.resize(0);
        builder.colIdxs.resize(0);
        builder.values.resize(0);
    }

    Matrix(
        Dim dim, Field<PetscInt> rowIdxs, Field<PetscInt> colIdxs, const Field<PetscScalar>& values
    )
    {
        MatCreate(MPI_COMM_WORLD, &mat);
        MatSetType(mat, MATSEQAIJKOKKOS);
        MatSetSizes(mat, dim.numRows, dim.numCols, dim.numRows, dim.numCols);
        MatSetPreallocationCOO(mat, rowIdxs.size(), rowIdxs.data(), colIdxs.data());
        MatSetValuesCOO(mat, values.data(), INSERT_VALUES);
    }

    void apply(const Field<PetscScalar>& in, Field<PetscScalar>& out)
    {
        Vec petscIn;
        Vec petscOut;

        VecCreateSeqKokkosWithArray(MPI_COMM_WORLD, 1, in.size(), in.data(), &petscIn);
        VecCreateSeqKokkosWithArray(MPI_COMM_WORLD, 1, out.size(), out.data(), &petscOut);

        MatMult(mat, petscIn, petscOut);
    }

    Executor get_executor() const { return GPUExecutor {}; }

    Mat getUnderlying() { return mat; }

private:

    Mat mat;
};

}
