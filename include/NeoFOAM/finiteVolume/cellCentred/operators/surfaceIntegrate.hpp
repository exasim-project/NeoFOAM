// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/sparsityPattern.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
void surfaceIntegrate(
    const Executor& exec,
    size_t nInternalFaces,
    std::span<const int> neighbour,
    std::span<const int> owner,
    std::span<const int> faceCells,
    std::span<const ValueType> flux,
    std::span<const scalar> V,
    std::span<ValueType> res,
    const dsl::Coeff operatorScaling
)
{
    size_t nCells {V.size()};
    const size_t nBoundaryFaces = faceCells.size();
    // check if the executor is GPU
    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            res[static_cast<size_t>(owner[i])] += flux[i];
            res[static_cast<size_t>(neighbour[i])] -= flux[i];
        }

        for (size_t i = nInternalFaces; i < nInternalFaces + nBoundaryFaces; i++)
        {
            auto own = static_cast<size_t>(faceCells[i - nInternalFaces]);
            res[own] += flux[i];
        }

        // TODO does it make sense to store invVol and multiply?
        for (size_t celli = 0; celli < nCells; celli++)
        {
            res[celli] *= operatorScaling[celli] / V[celli];
        }
    }
    else
    {
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
            KOKKOS_LAMBDA(const size_t celli) { res[celli] *= operatorScaling[celli] / V[celli]; }
        );
    }
}


template<typename ValueType>
class SurfaceIntegrate
{

public:

    SurfaceIntegrate(const SurfaceField<ValueType>& flux)
        : flux_(flux), type_(dsl::Operator::Type::Explicit), coeffs_(1.0) {};


    void build(const Input& input)
    {
        // do nothing
    }

    void explicitOperation(Field<ValueType>& source)
    {
        NeoFOAM::Field<NeoFOAM::scalar> tmpsource(source.exec(), source.size(), 0.0);
        const auto operatorScaling = this->getCoefficient();

        const UnstructuredMesh& mesh = flux_.mesh();
        const auto exec = flux_.exec();

        size_t nInternalFaces = mesh.nInternalFaces();
        surfaceIntegrate<ValueType>(
            exec,
            nInternalFaces,
            mesh.faceNeighbour().span(),
            mesh.faceOwner().span(),
            mesh.boundaryMesh().faceCells().span(),
            flux_.internalField().span(),
            mesh.cellVolumes().span(),
            tmpsource.span(),
            operatorScaling

        );
        source += tmpsource;
    }

    const Executor& exec() const { return flux_.exec(); }

    dsl::Coeff& getCoefficient() { return coeffs_; }

    const dsl::Coeff& getCoefficient() const { return coeffs_; }

    dsl::Operator::Type getType() const { return type_; }

    std::string getName() const { return "SurfaceIntegrate"; }

private:

    const SurfaceField<ValueType>& flux_;
    dsl::Operator::Type type_;
    dsl::Coeff coeffs_;
};


} // namespace NeoFOAM
