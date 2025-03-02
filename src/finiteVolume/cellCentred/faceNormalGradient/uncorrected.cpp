// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

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

Uncorrected::Uncorrected(
    const Executor& exec, const UnstructuredMesh& mesh, [[maybe_unused]] Input input
)
    : FaceNormalGradientFactory<scalar>::Register<Uncorrected>(exec, mesh),
      geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

Uncorrected::Uncorrected(const Executor& exec, const UnstructuredMesh& mesh)
    : FaceNormalGradientFactory<scalar>::Register<Uncorrected>(exec, mesh),
      geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

void Uncorrected::faceNormalGrad(
    const VolumeField<scalar>& volField, SurfaceField<scalar>& surfaceField
) const
{
    computeFaceNormalGrad(volField, geometryScheme_, surfaceField);
}

const SurfaceField<scalar>& Uncorrected::deltaCoeffs() const
{
    return geometryScheme_->nonOrthDeltaCoeffs();
}


std::unique_ptr<FaceNormalGradientFactory<scalar>> Uncorrected::clone() const
{
    return std::make_unique<Uncorrected>(*this);
}

} // namespace NeoFOAM
