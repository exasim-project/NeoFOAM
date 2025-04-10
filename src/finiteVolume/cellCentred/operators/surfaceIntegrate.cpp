// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors


#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void surfaceIntegrate(
    const Executor& exec,
    size_t nInternalFaces,
    std::span<const int> neighbour,
    std::span<const int> owner,
    std::span<const int> faceCells,
    std::span<const ValueType> flux,
    std::span<const scalar> v,
    std::span<ValueType> res,
    const dsl::Coeff operatorScaling
)
{
    size_t nCells {v.size()};
    const size_t nBoundaryFaces = faceCells.size();
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t i) {
            Kokkos::atomic_add(&res[static_cast<size_t>(owner[i])], flux[i]);
            Kokkos::atomic_sub(&res[static_cast<size_t>(neighbour[i])], flux[i]);
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + nBoundaryFaces},
        KOKKOS_LAMBDA(const size_t i) {
            auto own = static_cast<size_t>(faceCells[i - nInternalFaces]);
            Kokkos::atomic_add(&res[own], flux[i]);
        }
    );

    parallelFor(
        exec,
        {0, nCells},
        KOKKOS_LAMBDA(const size_t celli) { res[celli] *= operatorScaling[celli] / v[celli]; }
    );
}

#define NF_DECLARE_COMPUTE_IMP_INT(TYPENAME)                                                       \
    template void surfaceIntegrate<TYPENAME>(                                                      \
        const Executor&,                                                                           \
        size_t,                                                                                    \
        std::span<const int>,                                                                      \
        std::span<const int>,                                                                      \
        std::span<const int>,                                                                      \
        std::span<const TYPENAME>,                                                                 \
        std::span<const scalar>,                                                                   \
        std::span<TYPENAME>,                                                                       \
        const dsl::Coeff                                                                           \
    )

NF_DECLARE_COMPUTE_IMP_INT(scalar);
NF_DECLARE_COMPUTE_IMP_INT(Vector);

// instantiate the template class
template class SurfaceIntegrate<scalar>;
template class SurfaceIntegrate<Vector>;

};
