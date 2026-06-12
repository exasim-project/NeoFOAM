// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: BSD-3-Clause
//
// CUDA kernels for the aDIC preconditioner (see adic.hpp). One thread per row; every sweep is a
// gather (each thread writes only its own row entry), so there are no atomics. Launched on the
// Ginkgo executor's stream so they are ordered with the rest of the solve.
//
// HIP: this file ports verbatim -- replace <<<...>>> with the hip launch (or hipify) and CUstream_st
// with GKO_HIP_STREAM_STRUCT. SYCL/DPC++ is an analogous parallel_for.

#include <cstddef>

#include <ginkgo/ginkgo.hpp>

namespace gko_adic
{

namespace
{

template <typename ValueType, typename IndexType>
__global__ void extract_diag_kernel(
    std::size_t n, const ValueType* vals, const IndexType* col, const IndexType* row, ValueType* diag
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ValueType d = ValueType{0};
    for (auto k = row[i]; k < row[i + 1]; ++k) {
        if (static_cast<std::size_t>(col[k]) == i) {
            d = vals[k];
            break;
        }
    }
    diag[i] = d;
}

// rd[i] = 1 / ( diag[i] - sum_{j<i} A(i,j)^2 / diag[j] )   (gather over lower neighbours)
template <typename ValueType, typename IndexType>
__global__ void compute_rd_kernel(
    std::size_t n, const ValueType* vals, const IndexType* col, const IndexType* row,
    const ValueType* diag, ValueType* rd
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ValueType s = diag[i];
    for (auto k = row[i]; k < row[i + 1]; ++k) {
        const std::size_t j = static_cast<std::size_t>(col[k]);
        if (j < i) {
            s -= vals[k] * vals[k] / diag[j];
        }
    }
    rd[i] = ValueType{1} / s;
}

template <typename ValueType>
__global__ void diag_scale_kernel(
    std::size_t n, const ValueType* rd, const ValueType* b, ValueType* x
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = rd[i] * b[i];
    }
}

// work[i] = x[i] - rd[i] * sum_{j<i} A(i,j) * x[j]    (forward sweep, gather over lower neighbours)
template <typename ValueType, typename IndexType>
__global__ void forward_kernel(
    std::size_t n, const ValueType* vals, const IndexType* col, const IndexType* row,
    const ValueType* rd, const ValueType* x, ValueType* work
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ValueType s = x[i];
    for (auto k = row[i]; k < row[i + 1]; ++k) {
        const std::size_t j = static_cast<std::size_t>(col[k]);
        if (j < i) {
            s -= rd[i] * vals[k] * x[j];
        }
    }
    work[i] = s;
}

// x[i] = work[i] - rd[i] * sum_{j>i} A(i,j) * work[j]  (backward sweep, gather over upper neighbours)
template <typename ValueType, typename IndexType>
__global__ void backward_kernel(
    std::size_t n, const ValueType* vals, const IndexType* col, const IndexType* row,
    const ValueType* rd, ValueType* x, const ValueType* work
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ValueType s = work[i];
    for (auto k = row[i]; k < row[i + 1]; ++k) {
        const std::size_t j = static_cast<std::size_t>(col[k]);
        if (j > i) {
            s -= rd[i] * vals[k] * work[j];
        }
    }
    x[i] = s;
}

} // namespace


template <typename ValueType, typename IndexType>
void adic_generate_cuda(
    std::size_t n, const ValueType* vals, const IndexType* col, const IndexType* row,
    ValueType* diag, ValueType* rd, CUstream_st* stream
)
{
    if (n == 0) {
        return;
    }
    constexpr int block_size = 256;
    const auto grid_size = static_cast<unsigned>((n + block_size - 1) / block_size);
    extract_diag_kernel<<<grid_size, block_size, 0, stream>>>(n, vals, col, row, diag);
    compute_rd_kernel<<<grid_size, block_size, 0, stream>>>(n, vals, col, row, diag, rd);
}

template <typename ValueType, typename IndexType>
void adic_apply_cuda(
    std::size_t n, const ValueType* vals, const IndexType* col, const IndexType* row,
    const ValueType* rd, const ValueType* b, ValueType* x, ValueType* work, CUstream_st* stream
)
{
    if (n == 0) {
        return;
    }
    constexpr int block_size = 256;
    const auto grid_size = static_cast<unsigned>((n + block_size - 1) / block_size);
    diag_scale_kernel<<<grid_size, block_size, 0, stream>>>(n, rd, b, x);
    forward_kernel<<<grid_size, block_size, 0, stream>>>(n, vals, col, row, rd, x, work);
    backward_kernel<<<grid_size, block_size, 0, stream>>>(n, vals, col, row, rd, x, work);
}


#define GKO_ADIC_INSTANTIATE(ValueType, IndexType)                                              \
    template void adic_generate_cuda<ValueType, IndexType>(                                     \
        std::size_t, const ValueType*, const IndexType*, const IndexType*, ValueType*,          \
        ValueType*, CUstream_st*                                                                \
    );                                                                                          \
    template void adic_apply_cuda<ValueType, IndexType>(                                        \
        std::size_t, const ValueType*, const IndexType*, const IndexType*, const ValueType*,    \
        const ValueType*, ValueType*, ValueType*, CUstream_st*                                  \
    )

GKO_ADIC_INSTANTIATE(double, gko::int32);
GKO_ADIC_INSTANTIATE(float, gko::int32);
GKO_ADIC_INSTANTIATE(double, gko::int64);
GKO_ADIC_INSTANTIATE(float, gko::int64);

#undef GKO_ADIC_INSTANTIATE

} // namespace gko_adic
