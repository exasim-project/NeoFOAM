// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoFOAM::finiteVolume::cellCentred
{


void computeFaceNormalGrad(
    const VolumeField<scalar>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    SurfaceField<scalar>& surfaceField
)
{
    const UnstructuredMesh& mesh = surfaceField.mesh();
    const auto& exec = surfaceField.exec();

    const auto [owner, neighbour, surfFaceCells] =
        spans(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());


    const auto [phif, phi, phiBCValue, nonOrthDeltaCoeffs] = spans(
        surfaceField.internalField(),
        volField.internalField(),
        volField.boundaryField().value(),
        geometryScheme->nonOrthDeltaCoeffs().internalField()
    );

    size_t nInternalFaces = mesh.nInternalFaces();

    NeoFOAM::parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            phif[facei] = nonOrthDeltaCoeffs[facei] * (phi[neighbour[facei]] - phi[owner[facei]]);
        }
    );

    NeoFOAM::parallelFor(
        exec,
        {nInternalFaces, phif.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            auto faceBCI = facei - nInternalFaces;
            auto own = static_cast<size_t>(surfFaceCells[faceBCI]);

            phif[facei] = nonOrthDeltaCoeffs[facei] * (phiBCValue[faceBCI] - phi[own]);
        }
    );
}

template<typename ValueType>
class Uncorrected :
    public FaceNormalGradientFactory<ValueType>::template Register<Uncorrected<ValueType>>
{
    using Base = FaceNormalGradientFactory<ValueType>::template Register<Uncorrected<ValueType>>;


public:

    Uncorrected(const Executor& exec, const UnstructuredMesh& mesh, Input input)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    Uncorrected(const Executor& exec, const UnstructuredMesh& mesh)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    static std::string name() { return "uncorrected"; }

    static std::string doc() { return "Uncorrected interpolation"; }

    static std::string schema() { return "none"; }

    virtual void faceNormalGrad(
        const VolumeField<scalar>& volField, SurfaceField<scalar>& surfaceField
    ) const override
    {
        computeFaceNormalGrad(volField, geometryScheme_, surfaceField);
    }

    virtual const SurfaceField<scalar>& deltaCoeffs() const override
    {
        return geometryScheme_->nonOrthDeltaCoeffs();
    }

    std::unique_ptr<FaceNormalGradientFactory<ValueType>> clone() const override
    {
        return std::make_unique<Uncorrected>(*this);
    }

private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoFOAM
