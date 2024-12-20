// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

void computeLinearInterpolation(
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
    const auto sVolField = volField.internalField().span();
    const auto sBField = volField.boundaryField().value().span();
    const auto sOwner = owner.span();
    const auto sNeighbour = neighbour.span();
    size_t nInternalFaces = mesh.nInternalFaces();

    NeoFOAM::parallelFor(
        exec,
        {0, sfield.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            size_t own = static_cast<size_t>(sOwner[facei]);
            size_t nei = static_cast<size_t>(sNeighbour[facei]);
            if (facei < nInternalFaces)
            {
                sfield[facei] =
                    sWeight[facei] * sVolField[own] + (1 - sWeight[facei]) * sVolField[nei];
            }
            else
            {
                sfield[facei] = sWeight[facei] * sBField[facei - nInternalFaces];
            }
        }
    );
}

Linear::Linear(const Executor& exec, const UnstructuredMesh& mesh, [[maybe_unused]] Input input)
    : SurfaceInterpolationFactory::Register<Linear>(exec, mesh),
      geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

Linear::Linear(const Executor& exec, const UnstructuredMesh& mesh)
    : SurfaceInterpolationFactory::Register<Linear>(exec, mesh),
      geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

void Linear::interpolate(const VolumeField<scalar>& volField, SurfaceField<scalar>& surfaceField)
    const
{
    computeLinearInterpolation(volField, geometryScheme_, surfaceField);
}

void Linear::interpolate(
    [[maybe_unused]] const SurfaceField<scalar>& faceFlux,
    const VolumeField<scalar>& volField,
    SurfaceField<scalar>& surfaceField
) const
{
    interpolate(volField, surfaceField);
}

std::unique_ptr<SurfaceInterpolationFactory> Linear::clone() const
{
    return std::make_unique<Linear>(*this);
}

} // namespace NeoFOAM
