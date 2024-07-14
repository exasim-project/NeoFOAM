// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/interpolation/upwind.hpp"
#include <memory>
#include "NeoFOAM/core/error.hpp"

namespace NeoFOAM
{

Upwind::Upwind(const Executor& exec, const UnstructuredMesh& mesh)
    : SurfaceInterpolationKernel(exec, mesh), mesh_(mesh),
      geometryScheme_(FvccGeometryScheme::readOrCreate(mesh)) {

      };


void Upwind::interpolate(
    const GPUExecutor& exec,
    fvcc::SurfaceField<scalar>& surfaceField,
    const fvcc::VolumeField<scalar>& volField
)
{
    NF_ERROR_EXIT("limited scheme require a faceFlux");
}

void Upwind::interpolate(
    const OMPExecutor& exec,
    fvcc::SurfaceField<scalar>& surfaceField,
    const fvcc::VolumeField<scalar>& volField
)
{
    NF_ERROR_EXIT("limited scheme require a faceFlux");
}

void Upwind::interpolate(
    const CPUExecutor& exec,
    fvcc::SurfaceField<scalar>& surfaceField,
    const fvcc::VolumeField<scalar>& volField
)
{
    NF_ERROR_EXIT("limited scheme require a faceFlux");
}

void Upwind::interpolate(
    const GPUExecutor& exec,
    fvcc::SurfaceField<scalar>& surfaceField,
    const fvcc::SurfaceField<scalar>& faceFlux,
    const fvcc::VolumeField<scalar>& volField
)
{
    using executor = typename GPUExecutor::exec;
    auto sfield = surfaceField.internalField().span();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();

    const auto s_weight = geometryScheme_->weights().internalField().span();
    const auto s_faceFlux = faceFlux.internalField().span();
    const auto s_volField = volField.internalField().span();
    const auto s_bField = volField.boundaryField().value().span();
    const auto s_owner = owner.span();
    const auto s_neighbour = neighbour.span();
    int nInternalFaces = mesh_.nInternalFaces();
    Kokkos::parallel_for(
        "gaussGreenGrad",
        Kokkos::RangePolicy<executor>(0, sfield.size()),
        KOKKOS_LAMBDA(const int facei) {
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


void Upwind::interpolate(
    const OMPExecutor& exec,
    fvcc::SurfaceField<scalar>& surfaceField,
    const fvcc::SurfaceField<scalar>& faceFlux,
    const fvcc::VolumeField<scalar>& volField
)
{
    using executor = typename OMPExecutor::exec;
    auto sfield = surfaceField.internalField().span();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();

    const auto s_weight = geometryScheme_->weights().internalField().span();
    const auto s_faceFlux = faceFlux.internalField().span();
    const auto s_volField = volField.internalField().span();
    const auto s_bField = volField.boundaryField().value().span();
    const auto s_owner = owner.span();
    const auto s_neighbour = neighbour.span();
    int nInternalFaces = mesh_.nInternalFaces();
    Kokkos::parallel_for(
        "gaussGreenGrad",
        Kokkos::RangePolicy<executor>(0, sfield.size()),
        KOKKOS_LAMBDA(const int facei) {
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

void Upwind::interpolate(
    const CPUExecutor& exec,
    fvcc::SurfaceField<scalar>& surfaceField,
    const fvcc::SurfaceField<scalar>& faceFlux,
    const fvcc::VolumeField<scalar>& volField
)
{
    using executor = typename CPUExecutor::exec;
    auto sfield = surfaceField.internalField().span();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();
    const auto s_weight = geometryScheme_->weights().internalField().span();
    const auto s_faceFlux = faceFlux.internalField().span();
    const auto s_volField = volField.internalField().span();
    const auto s_bField = volField.boundaryField().value().span();
    const auto s_owner = owner.span();
    const auto s_neighbour = neighbour.span();
    int nInternalFaces = mesh_.nInternalFaces();
    for (size_t facei = 0; facei < sfield.size(); facei++)
    {
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
}


std::unique_ptr<SurfaceInterpolationKernel> Upwind::clone() const
{
    return std::make_unique<Upwind>(exec_, mesh_);
}


} // namespace NeoFOAM
