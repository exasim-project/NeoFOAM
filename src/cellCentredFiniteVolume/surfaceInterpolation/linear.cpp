// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/linear.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolationFactory.hpp"
#include <memory>
#include "NeoFOAM/core/Error.hpp"

namespace NeoFOAM
{

// std::size_t linear::_index = surfaceInterpolationFactory::registerClass("linear", [](const executor& exec, const unstructuredMesh& mesh)
//                                                                         { return std::make_unique<linear>(exec, mesh); });

linear::linear(const executor& exec, const unstructuredMesh& mesh)
    : surfaceInterpolationKernel(exec, mesh),
      mesh_(mesh),
    //   geometryScheme_(std::make_shared<FvccGeometryScheme>(mesh))
      geometryScheme_(FvccGeometryScheme::readOrCreate(mesh))
{
    // if (geometryScheme_ == nullptr)
    // {
    //     error("geometryScheme_ is not initialized").exit();
    // }

};

void linear::operator()(const GPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField)
{
    using executor = typename GPUExecutor::exec;
    auto sfield = surfaceField.internalField().field();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();

    const auto s_weight = geometryScheme_->weights().internalField().field();    
    auto s_volField = volField.internalField().field();
    auto s_bField = volField.boundaryField().value().field();
    auto s_owner = owner.field();
    auto s_neighbour = neighbour.field();
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
                sfield[facei] = s_weight[facei]*s_bField[pfacei];
            }
        }
    );
}


void linear::operator()(const OMPExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField)
{
    using executor = typename OMPExecutor::exec;
    auto sfield = surfaceField.internalField().field();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();

    const auto s_weight = geometryScheme_->weights().internalField().field();
    auto s_volField = volField.internalField().field();
    auto s_bField = volField.boundaryField().value().field();
    auto s_owner = owner.field();
    auto s_neighbour = neighbour.field();
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
                sfield[facei] = s_weight[facei]*s_bField[pfacei];
            }
        }
    );
}

void linear::operator()(const CPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField)
{
    using executor = typename CPUExecutor::exec;
    auto sfield = surfaceField.internalField().field();
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();
    const auto s_weight = geometryScheme_->weights().internalField().field();
    auto s_volField = volField.internalField().field();
    auto s_bField = volField.boundaryField().value().field();
    auto s_owner = owner.field();
    auto s_neighbour = neighbour.field();
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
                sfield[facei] = s_weight[facei]*s_bField[pfacei];
            }
        }
    );
}


// surfaceInterpolationFactory::registerClass("linear", [](const NeoFOAM::executor& exec, const NeoFOAM::unstructuredMesh& mesh) {
//     return std::make_unique<NeoFOAM::linear>(exec, mesh);
// });

} // namespace NeoFOAM