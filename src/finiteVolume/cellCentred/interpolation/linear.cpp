// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

void computeLinearInterpolation(
    SurfaceField<scalar>& surfaceField,
    const VolumeField<scalar>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme
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
    int nInternalFaces = mesh.nInternalFaces();

    NeoFOAM::parallelFor(
        exec,
        {0, sfield.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            int32_t own = sOwner[facei];
            int32_t nei = sNeighbour[facei];
            if (facei < nInternalFaces)
            {
                sfield[facei] =
                    sWeight[facei] * sVolField[own] + (1 - sWeight[facei]) * sVolField[nei];
            }
            else
            {
                int pfacei = facei - nInternalFaces;
                sfield[facei] = sWeight[facei] * sBField[pfacei];
            }
        }
    );
}

Linear::Linear(const Executor& exec, const UnstructuredMesh& mesh)
    : SurfaceInterpolationFactory::Register<Linear>(exec, mesh),
      geometryScheme_(GeometryScheme::readOrCreate(mesh)) {

      };

void Linear::interpolate(SurfaceField<scalar>& surfaceField, const VolumeField<scalar>& volField)
{
    computeLinearInterpolation(surfaceField, volField, geometryScheme_);
}

void Linear::interpolate(
    SurfaceField<scalar>& surfaceField,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<scalar>& volField
)
{
    interpolate(surfaceField, volField);
}

std::unique_ptr<SurfaceInterpolationFactory> Linear::clone() const
{
    return std::make_unique<Linear>(*this);
}


} // namespace NeoFOAM
