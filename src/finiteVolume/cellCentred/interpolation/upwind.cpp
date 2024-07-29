// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/interpolation/upwind.hpp"
#include "NeoFOAM/core/error.hpp"
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
    int nInternalFaces = mesh.nInternalFaces();

    NeoFOAM::parallelFor(
        exec,
        {0, sfield.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            int32_t own = sOwner[facei];
            int32_t nei = sNeighbour[facei];
            if (facei < nInternalFaces)
            {
                if (sFaceFlux[facei] >= 0)
                {
                    sfield[facei] = sVolField[own];
                }
                else
                {
                    sfield[facei] = sVolField[nei];
                }
            }
            else
            {
                int pfacei = facei - nInternalFaces;
                sfield[facei] = sWeight[facei] * sBField[pfacei];
            }
        }
    );
}

Upwind::Upwind(const Executor& exec, const UnstructuredMesh& mesh)
    : SurfaceInterpolationFactory::Register<Upwind>(exec, mesh),
      geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};


void Upwind::interpolate(const VolumeField<scalar>& volField, SurfaceField<scalar>& surfaceField)
{
    NF_ERROR_EXIT("limited scheme require a faceFlux");
}

void Upwind::interpolate(
    SurfaceField<scalar>& surfaceField,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<scalar>& volField
)
{
    computeUpwindInterpolation(faceFlux, volField, geometryScheme_, surfaceField);
}

std::unique_ptr<SurfaceInterpolationFactory> Upwind::clone() const
{
    return std::make_unique<Upwind>(*this);
}


} // namespace NeoFOAM
