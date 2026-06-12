// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: BSD-3-Clause
//
// aDIC ("approximate DIC") preconditioner as a native Ginkgo LinOp.
//
// This is OpenFOAM/SPUMA's approximate diagonal-based incomplete-Cholesky preconditioner,
// expressed in Ginkgo's execution model so that the apply runs on Ginkgo's own executor/stream
// (no cross-framework synchronisation). The standard DIC triangular solves -- which are sequential
// recurrences -- are replaced by a single forward and a single backward Jacobi-style sweep.
//
// Because a CSR matrix stores both A(i,j) and A(j,i), and aDIC targets *symmetric* matrices
// (the pressure Laplacian), every sweep is a GATHER over each row: row i reads its neighbours and
// writes only its own entry. There are therefore no atomics and no scatter conflicts -- just three
// embarrassingly parallel passes per apply.
//
// Intended for use as a `generated_preconditioner` of a Ginkgo Krylov solver:
//
//     auto adic = gko::share(Adic<double, int>::create(exec, csr));
//     auto cg   = gko::solver::Cg<double>::build()
//                     .with_generated_preconditioner(adic)
//                     .with_criteria(...)
//                     .on(exec)->generate(csr);
//
// Caveats / upstreaming notes:
//   * Symmetric, positive-definite matrices only (hand it (-A)x=(-b) for the negative-definite
//     OpenFOAM pressure Laplacian).
//   * Single right-hand side (stride-1 Dense). Multi-RHS is a straightforward extension.
//   * HIP/SYCL kernels are not provided here (CUDA + host only); they are trivial ports of the
//     CUDA kernel. For a full Ginkgo PR the run() bodies move into the modular kernel files
//     (reference/, omp/, common/cuda_hip/, dpcpp/) and a Factory + config registration are added,
//     modelled on gko::preconditioner::Ic.

#pragma once

#include <memory>

#include <ginkgo/ginkgo.hpp>

namespace gko_adic
{

// CUDA kernel launchers (defined in adic_kernels.cu). Declared here so the host translation unit
// does not need to be compiled by nvcc. `stream` is the Ginkgo executor's CUDA stream
// (CUstream_st* is Ginkgo's forward-declared CUstream), so the kernels run on the same stream as
// the rest of the solver -- ordered with no explicit synchronisation.
template <typename ValueType, typename IndexType>
void adic_generate_cuda(
    std::size_t num_rows, const ValueType* vals, const IndexType* col_idxs,
    const IndexType* row_ptrs, ValueType* diag, ValueType* rd, CUstream_st* stream
);

template <typename ValueType, typename IndexType>
void adic_apply_cuda(
    std::size_t num_rows, const ValueType* vals, const IndexType* col_idxs,
    const IndexType* row_ptrs, const ValueType* rd, const ValueType* b, ValueType* x,
    ValueType* work, CUstream_st* stream
);


namespace detail
{

// rd[i] = 1 / ( diag[i] - sum_{j<i, A(i,j)!=0} A(i,j)^2 / diag[j] )
// The row-i entries with column index < i are exactly the lower neighbours, so this is a gather.
// `diag` is a scratch buffer holding the ORIGINAL diagonal (read-only in pass 2) -- kept separate
// from `rd` to avoid the read-after-write hazard of writing rd[i] while another row reads it.
template <typename ValueType, typename IndexType>
void adic_generate_host(
    std::size_t num_rows, const ValueType* vals, const IndexType* col_idxs,
    const IndexType* row_ptrs, ValueType* diag, ValueType* rd
)
{
#pragma omp parallel for
    for (std::size_t i = 0; i < num_rows; ++i) {
        ValueType d = gko::zero<ValueType>();
        for (auto k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
            if (static_cast<std::size_t>(col_idxs[k]) == i) {
                d = vals[k];
                break;
            }
        }
        diag[i] = d;
    }
#pragma omp parallel for
    for (std::size_t i = 0; i < num_rows; ++i) {
        ValueType s = diag[i];
        for (auto k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
            const auto j = static_cast<std::size_t>(col_idxs[k]);
            if (j < i) {
                s -= vals[k] * vals[k] / diag[j];
            }
        }
        rd[i] = gko::one<ValueType>() / s;
    }
}

// x = M^{-1} b, with M the aDIC factor. Three gathers:
//   1. diagonal scale:   x[i] = rd[i] * b[i]
//   2. forward sweep:    work[i] = x[i]    - rd[i] * sum_{j<i} A(i,j) * x[j]
//   3. backward sweep:   x[i]    = work[i] - rd[i] * sum_{j>i} A(i,j) * work[j]
template <typename ValueType, typename IndexType>
void adic_apply_host(
    std::size_t num_rows, const ValueType* vals, const IndexType* col_idxs,
    const IndexType* row_ptrs, const ValueType* rd, const ValueType* b, ValueType* x,
    ValueType* work
)
{
#pragma omp parallel for
    for (std::size_t i = 0; i < num_rows; ++i) {
        x[i] = rd[i] * b[i];
    }
#pragma omp parallel for
    for (std::size_t i = 0; i < num_rows; ++i) {
        ValueType s = x[i];
        for (auto k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
            const auto j = static_cast<std::size_t>(col_idxs[k]);
            if (j < i) {
                s -= rd[i] * vals[k] * x[j];
            }
        }
        work[i] = s;
    }
#pragma omp parallel for
    for (std::size_t i = 0; i < num_rows; ++i) {
        ValueType s = work[i];
        for (auto k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
            const auto j = static_cast<std::size_t>(col_idxs[k]);
            if (j > i) {
                s -= rd[i] * vals[k] * work[j];
            }
        }
        x[i] = s;
    }
}

} // namespace detail


template <typename ValueType = double, typename IndexType = gko::int32>
class Adic : public gko::EnableLinOp<Adic<ValueType, IndexType>>,
             public gko::EnableCreateMethod<Adic<ValueType, IndexType>> {
    friend class gko::EnablePolymorphicObject<Adic, gko::LinOp>;
    friend class gko::EnableCreateMethod<Adic>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Dense = gko::matrix::Dense<ValueType>;

    // Build from a (symmetric, positive-definite) CSR matrix: copies nothing, but computes and
    // stores the reciprocal preconditioned diagonal `rd_`.
    Adic(std::shared_ptr<const gko::Executor> exec, std::shared_ptr<const Csr> mtx)
        : gko::EnableLinOp<Adic>(exec, mtx ? mtx->get_size() : gko::dim<2>{}),
          mtx_{std::move(mtx)},
          rd_{exec, mtx_ ? mtx_->get_size()[0] : 0},
          work_{exec, mtx_ ? mtx_->get_size()[0] : 0}
    {
        if (mtx_) {
            generate();
        }
    }

protected:
    // Used only by Ginkgo's polymorphic-object machinery.
    explicit Adic(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<Adic>(exec), rd_{exec}, work_{exec}
    {}

    void generate()
    {
        const auto n = mtx_->get_size()[0];
        gko::array<ValueType> diag{this->get_executor(), n};

        struct generate_op : gko::Operation {
            std::size_t n;
            const ValueType* vals;
            const IndexType* col;
            const IndexType* row;
            ValueType* diag;
            ValueType* rd;

            void run(std::shared_ptr<const gko::OmpExecutor>) const override
            {
                detail::adic_generate_host(n, vals, col, row, diag, rd);
            }
            void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
            {
                detail::adic_generate_host(n, vals, col, row, diag, rd);
            }
            void run(std::shared_ptr<const gko::CudaExecutor> exec) const override
            {
                adic_generate_cuda(n, vals, col, row, diag, rd, exec->get_stream());
            }
        };

        this->get_executor()->run(generate_op{
            n, mtx_->get_const_values(), mtx_->get_const_col_idxs(), mtx_->get_const_row_ptrs(),
            diag.get_data(), rd_.get_data()});
    }

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<Dense>(b);
        auto dense_x = gko::as<Dense>(x);
        const auto n = mtx_->get_size()[0];

        struct apply_op : gko::Operation {
            std::size_t n;
            const ValueType* vals;
            const IndexType* col;
            const IndexType* row;
            const ValueType* rd;
            const ValueType* b;
            ValueType* x;
            ValueType* work;

            void run(std::shared_ptr<const gko::OmpExecutor>) const override
            {
                detail::adic_apply_host(n, vals, col, row, rd, b, x, work);
            }
            void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
            {
                detail::adic_apply_host(n, vals, col, row, rd, b, x, work);
            }
            void run(std::shared_ptr<const gko::CudaExecutor> exec) const override
            {
                adic_apply_cuda(n, vals, col, row, rd, b, x, work, exec->get_stream());
            }
        };

        this->get_executor()->run(apply_op{
            n, mtx_->get_const_values(), mtx_->get_const_col_idxs(), mtx_->get_const_row_ptrs(),
            rd_.get_const_data(), dense_b->get_const_values(), dense_x->get_values(),
            work_.get_data()});
    }

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        auto dense_x = gko::as<Dense>(x);
        auto tmp = dense_x->clone();
        this->apply_impl(b, tmp.get());
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, tmp);
    }

private:
    std::shared_ptr<const Csr> mtx_;
    gko::array<ValueType> rd_;          // reciprocal preconditioned diagonal
    mutable gko::array<ValueType> work_; // apply scratch (forward-sweep result)
};

} // namespace gko_adic
