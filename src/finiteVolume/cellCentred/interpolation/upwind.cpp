// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/upwind.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolationSelector.hpp"
#include <memory>
#include "NeoFOAM/core/Error.hpp"

namespace NeoFOAM
{

upwind::upwind(const executor& exec, const UnstructuredMesh& mesh)
    : SurfaceInterpolationKernel(exec, mesh), mesh_(mesh),
      geometryScheme_(FvccGeometryScheme::readOrCreate(mesh)) {

      };


void upwind::interpolate(
    const GPUExecutor& exec,
    fvccSurfaceField<scalar>& surfaceField,
    const fvccVolField<scalar>& volField
)
{
    error("limited scheme require a faceFlux").exit();
}

void upwind::interpolate(
    const OMPExecutor& exec,
    fvccSurfaceField<scalar>& surfaceField,
    const fvccVolField<scalar>& volField
)
{
    error("limited scheme require a faceFlux").exit();
}

void upwind::interpolate(
    const CPUExecutor& exec,
    fvccSurfaceField<scalar>& surfaceField,
    const fvccVolField<scalar>& volField
)
{
    error("limited scheme require a faceFlux").exit();
}

void upwind::interpolate(
    const GPUExecutor& exec,
    fvccSurfaceField<scalar>& surfaceField,
    const fvccSurfaceField<scalar>& faceFlux,
    const fvccVolField<scalar>& volField
)
{
    using executor = typename GPUExecutor::exec;
    auto sfield = surfaceField.internalField().field();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();

    const auto s_weight = geometryScheme_->weights().internalField().field();
    const auto s_faceFlux = faceFlux.internalField().field();
    const auto s_volField = volField.internalField().field();
    const auto s_bField = volField.boundaryField().value().field();
    const auto s_owner = owner.field();
    const auto s_neighbour = neighbour.field();
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


void upwind::interpolate(
    const OMPExecutor& exec,
    fvccSurfaceField<scalar>& surfaceField,
    const fvccSurfaceField<scalar>& faceFlux,
    const fvccVolField<scalar>& volField
)
{
    using executor = typename OMPExecutor::exec;
    auto sfield = surfaceField.internalField().field();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();

    const auto s_weight = geometryScheme_->weights().internalField().field();
    const auto s_faceFlux = faceFlux.internalField().field();
    const auto s_volField = volField.internalField().field();
    const auto s_bField = volField.boundaryField().value().field();
    const auto s_owner = owner.field();
    const auto s_neighbour = neighbour.field();
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

void upwind::interpolate(
    const CPUExecutor& exec,
    fvccSurfaceField<scalar>& surfaceField,
    const fvccSurfaceField<scalar>& faceFlux,
    const fvccVolField<scalar>& volField
)
{
    using executor = typename CPUExecutor::exec;
    auto sfield = surfaceField.internalField().field();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();
    const auto s_weight = geometryScheme_->weights().internalField().field();
    const auto s_faceFlux = faceFlux.internalField().field();
    const auto s_volField = volField.internalField().field();
    const auto s_bField = volField.boundaryField().value().field();
    const auto s_owner = owner.field();
    const auto s_neighbour = neighbour.field();
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


std::unique_ptr<SurfaceInterpolationKernel> upwind::clone() const
{
    return std::make_unique<upwind>(exec_, mesh_);
}

std::unique_ptr<SurfaceInterpolationKernel>
upwind::Create(const executor& exec, const UnstructuredMesh& mesh)
{
    return std::make_unique<upwind>(exec, mesh);
}

bool upwind::s_registered = CompressionMethodFactory::Register("upwind", upwind::Create);

} // namespace NeoFOAM
