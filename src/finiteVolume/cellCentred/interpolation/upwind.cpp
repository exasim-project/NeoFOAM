// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/interpolation/upwind.hpp"
#include <memory>
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
namespace NeoFOAM
{


void detail::computeUpwindInterpolation(
    fvcc::SurfaceField<scalar>& surfaceField,
    const fvcc::SurfaceField<scalar>& faceFlux,
    const fvcc::VolumeField<scalar>& volField,
    const std::shared_ptr<FvccGeometryScheme> geometryScheme
)
{
    const UnstructuredMesh& mesh = surfaceField.mesh();
    const auto& exec = surfaceField.exec();

    auto sfield = surfaceField.internalField().span();
    const NeoFOAM::labelField& owner = mesh.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh.faceNeighbour();
    const auto s_weight = geometryScheme->weights().internalField().span();
    const auto s_faceFlux = faceFlux.internalField().span();
    const auto s_volField = volField.internalField().span();
    const auto s_bField = volField.boundaryField().value().span();
    const auto s_owner = owner.span();
    const auto s_neighbour = neighbour.span();
    int nInternalFaces = mesh.nInternalFaces();

    NeoFOAM::parallelFor(
        exec,
        {0, sfield.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            int32_t own = s_owner[facei];
            int32_t nei = s_neighbour[facei];
            if (facei < nInternalFaces)
            {
                if (s_faceFlux[facei] >= 0)
                {
                    sfield[facei] = s_volField[own];
                }
                else
                {
                    sfield[facei] = s_volField[nei];
                }
            }
            else
            {
                int pfacei = facei - nInternalFaces;
                sfield[facei] = s_weight[facei] * s_bField[pfacei];
            }
        }
    );
}

Upwind::Upwind(const Executor& exec, const UnstructuredMesh& mesh)
    : SurfaceInterpolationFactory::Register<Upwind>(exec, mesh),
      geometryScheme_(FvccGeometryScheme::readOrCreate(mesh)) {

      };


void Upwind::interpolate(
    fvcc::SurfaceField<scalar>& surfaceField, const fvcc::VolumeField<scalar>& volField
)
{
    NF_ERROR_EXIT("limited scheme require a faceFlux");
}

void Upwind::interpolate(
    fvcc::SurfaceField<scalar>& surfaceField,
    const fvcc::SurfaceField<scalar>& faceFlux,
    const fvcc::VolumeField<scalar>& volField
)
{
    detail::computeUpwindInterpolation(surfaceField, faceFlux, volField, geometryScheme_);
}

std::unique_ptr<SurfaceInterpolationFactory> Upwind::clone() const
{
    return std::make_unique<Upwind>(*this);
}


} // namespace NeoFOAM
