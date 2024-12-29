// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/interpolation/upwind.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

void computeUpwindInterpolation(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<scalar>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    SurfaceField<scalar>& surfaceField
)
{
    const UnstructuredMesh& mesh = surfaceField.mesh();
    const auto& exec = surfaceField.exec();

    auto sfield = surfaceField.internalField().span();
    const NeoFOAM::labelField& owner = mesh.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh.faceNeighbour();
    const auto sWeight = geometryScheme->weights().internalField().span();
    const auto sFaceFlux = faceFlux.internalField().span();
    const auto sVolField = volField.internalField().span();
    const auto sBField = volField.boundaryField().value().span();
    const auto sOwner = owner.span();
    const auto sNeighbour = neighbour.span();
    size_t nInternalFaces = mesh.nInternalFaces();

    NeoFOAM::parallelFor(
        exec,
        {0, sfield.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            if (facei < nInternalFaces)
            {
                if (sFaceFlux[facei] >= 0)
                {
                    size_t own = static_cast<size_t>(sOwner[facei]);
                    sfield[facei] = sVolField[own];
                }
                else
                {
                    size_t nei = static_cast<size_t>(sNeighbour[facei]);
                    sfield[facei] = sVolField[nei];
                }
            }
            else
            {
                sfield[facei] = sWeight[facei] * sBField[facei - nInternalFaces];
            }
        }
    );
}

Upwind::Upwind(const Executor& exec, const UnstructuredMesh& mesh, [[maybe_unused]] Input input)
    : SurfaceInterpolationFactory::Register<Upwind>(exec, mesh),
      geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

void Upwind::interpolate(
    [[maybe_unused]] const VolumeField<scalar>& volField,
    [[maybe_unused]] SurfaceField<scalar>& surfaceField
) const
{
    NF_ERROR_EXIT("limited scheme require a faceFlux");
}

void Upwind::interpolate(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<scalar>& volField,
    SurfaceField<scalar>& surfaceField
) const
{
    computeUpwindInterpolation(faceFlux, volField, geometryScheme_, surfaceField);
}

std::unique_ptr<SurfaceInterpolationFactory> Upwind::clone() const
{
    return std::make_unique<Upwind>(*this);
}

} // namespace NeoFOAM
