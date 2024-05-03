// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/linear.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolationSelector.hpp"
#include <memory>
#include "NeoFOAM/core/Error.hpp"

namespace NeoFOAM
{

linear::linear(const executor& exec, const unstructuredMesh& mesh)
    : surfaceInterpolationKernel(exec, mesh),
      mesh_(mesh),
      geometryScheme_(FvccGeometryScheme::readOrCreate(mesh)) {

      };

void linear::interpolate(const GPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField)
{
    using executor = typename GPUExecutor::exec;
    auto sfield = surfaceField.internalField().field();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();

    const auto s_weight = geometryScheme_->weights().internalField().field();
    const auto s_volField = volField.internalField().field();
    const auto s_bField = volField.boundaryField().value().field();
    const auto s_owner = owner.field();
    const auto s_neighbour = neighbour.field();
    int nInternalFaces = mesh_.nInternalFaces();
    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, sfield.size()), KOKKOS_LAMBDA(const int facei) {
            int32_t own = s_owner[facei];
            int32_t nei = s_neighbour[facei];
            if (facei < nInternalFaces)
            {
                sfield[facei] = s_weight[facei] * s_volField[own] + (1 - s_weight[facei]) * s_volField[nei];
            }
            else
            {
                int pfacei = facei - nInternalFaces;
                sfield[facei] = s_weight[facei] * s_bField[pfacei];
            }
        }
    );
}


void linear::interpolate(const OMPExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField)
{
    using executor = typename OMPExecutor::exec;
    auto sfield = surfaceField.internalField().field();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();

    const auto s_weight = geometryScheme_->weights().internalField().field();
    const auto s_volField = volField.internalField().field();
    const auto s_bField = volField.boundaryField().value().field();
    const auto s_owner = owner.field();
    const auto s_neighbour = neighbour.field();
    int nInternalFaces = mesh_.nInternalFaces();
    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, sfield.size()), KOKKOS_LAMBDA(const int facei) {
            int32_t own = s_owner[facei];
            int32_t nei = s_neighbour[facei];
            if (facei < nInternalFaces)
            {
                sfield[facei] = s_weight[facei] * s_volField[own] + (1 - s_weight[facei]) * s_volField[nei];
            }
            else
            {
                int pfacei = facei - nInternalFaces;
                sfield[facei] = s_weight[facei] * s_bField[pfacei];
            }
        }
    );
}

void linear::interpolate(const CPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField)
{
    using executor = typename CPUExecutor::exec;
    auto sfield = surfaceField.internalField().field();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();

    const auto s_weight = geometryScheme_->weights().internalField().field();
    const auto s_volField = volField.internalField().field();
    const auto s_bField = volField.boundaryField().value().field();
    const auto s_owner = owner.field();
    const auto s_neighbour = neighbour.field();
    int nInternalFaces = mesh_.nInternalFaces();
    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, sfield.size()), KOKKOS_LAMBDA(const int facei) {
            int32_t own = s_owner[facei];
            int32_t nei = s_neighbour[facei];
            if (facei < nInternalFaces)
            {
                sfield[facei] = s_weight[facei] * s_volField[own] + (1 - s_weight[facei]) * s_volField[nei];
            }
            else
            {
                int pfacei = facei - nInternalFaces;
                sfield[facei] = s_weight[facei] * s_bField[pfacei];
            }
        }
    );
}

void linear::interpolate(const GPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& volField)
{
    interpolate(exec, surfaceField, volField);
}

void linear::interpolate(const OMPExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& volField)
{
    interpolate(exec, surfaceField, volField);
}
void linear::interpolate(const CPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& volField)
{
    interpolate(exec, surfaceField, volField);
}

std::unique_ptr<surfaceInterpolationKernel> linear::clone() const
{
    return std::make_unique<linear>(exec_, mesh_);
}

std::unique_ptr<surfaceInterpolationKernel> linear::Create(const executor& exec, const unstructuredMesh& mesh)
{
    return std::make_unique<linear>(exec, mesh);
}

bool linear::s_registered = CompressionMethodFactory::Register("linear", linear::Create);

} // namespace NeoFOAM
