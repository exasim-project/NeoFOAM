// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"

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
);

template<typename ValueType>
class SurfaceIntegrate
{

public:

    using FieldValueType = ValueType;

    SurfaceIntegrate(const SurfaceField<ValueType>& flux)
        : flux_(flux), type_(dsl::Operator::Type::Explicit), coeffs_(1.0) {};

    SurfaceIntegrate(const SurfaceIntegrate& surfaceIntegrate)
        : flux_(surfaceIntegrate.flux_), type_(surfaceIntegrate.type_),
          coeffs_(surfaceIntegrate.coeffs_) {};


    void build(const Input&) {}

    void explicitOperation(Field<ValueType>& source) const
    {
        NeoN::Field<ValueType> tmpsource(source.exec(), source.size(), zero<ValueType>());
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
            this->flux_.internalField().span(),
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


} // namespace NeoN
