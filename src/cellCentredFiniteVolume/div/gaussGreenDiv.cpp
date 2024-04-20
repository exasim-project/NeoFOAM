// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/div/gaussGreenDiv.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryFieldSelector.hpp"
#include <functional>

namespace NeoFOAM
{

GaussGreenDivKernel::GaussGreenDivKernel(const unstructuredMesh& mesh, const surfaceInterpolation& surfInterp)
    : mesh_(mesh), surfaceInterpolation_(surfInterp) {

                   };

void GaussGreenDivKernel::operator()(const GPUExecutor& exec, fvccVolField<scalar>& divPhi, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& phi)
{
    using executor = typename GPUExecutor::exec;
    fvccSurfaceField<NeoFOAM::scalar> phif(exec, mesh_, NeoFOAM::createCalculatedBCs<scalar>(mesh_));
    const auto s_faceCells = mesh_.boundaryMesh().faceCells().field();
    surfaceInterpolation_.interpolate(phif,faceFlux,phi);

    auto s_divPhi = divPhi.internalField().field();
    
    const auto s_phif = phif.internalField().field();
    const auto s_owner = mesh_.faceOwner().field();
    const auto s_neighbour = mesh_.faceNeighbour().field();
    const auto s_faceFlux = faceFlux.internalField().field();
    size_t nInternalFaces = mesh_.nInternalFaces();

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, nInternalFaces), KOKKOS_LAMBDA(const int i) {
            NeoFOAM::scalar Flux = s_faceFlux[i] * s_phif[i];
            Kokkos::atomic_add(&s_divPhi[s_owner[i]], Flux);
            Kokkos::atomic_sub(&s_divPhi[s_neighbour[i]], Flux);
        }
    );

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(nInternalFaces, s_phif.size()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_faceCells[i-nInternalFaces];
            NeoFOAM::scalar value_own = s_faceFlux[i] * s_phif[i];
            Kokkos::atomic_add(&s_divPhi[own], value_own);
        }
    );

    const auto s_V = mesh_.cellVolumes().field();
    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, mesh_.nCells()), KOKKOS_LAMBDA(const int celli) {
            s_divPhi[celli] *= 1/s_V[celli];
        }
    );
}

void GaussGreenDivKernel::operator()(const OMPExecutor& exec, fvccVolField<scalar>& divPhi, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& phi)
{
    using executor = typename OMPExecutor::exec;
    fvccSurfaceField<NeoFOAM::scalar> phif(exec, mesh_, NeoFOAM::createCalculatedBCs<scalar>(mesh_));
    const auto s_faceCells = mesh_.boundaryMesh().faceCells().field();
    surfaceInterpolation_.interpolate(phif,faceFlux,phi);

    auto s_divPhi = divPhi.internalField().field();
    
    const auto s_phif = phif.internalField().field();
    const auto s_owner = mesh_.faceOwner().field();
    const auto s_neighbour = mesh_.faceNeighbour().field();
    const auto s_faceFlux = faceFlux.internalField().field();
    size_t nInternalFaces = mesh_.nInternalFaces();

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, nInternalFaces), KOKKOS_LAMBDA(const int i) {
            NeoFOAM::scalar Flux = s_faceFlux[i] * s_phif[i];
            Kokkos::atomic_add(&s_divPhi[s_owner[i]], Flux);
            Kokkos::atomic_sub(&s_divPhi[s_neighbour[i]], Flux);
        }
    );

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(nInternalFaces, s_phif.size()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_faceCells[i-nInternalFaces];
            NeoFOAM::scalar value_own = s_faceFlux[i] * s_phif[i];
            Kokkos::atomic_add(&s_divPhi[own], value_own);
        }
    );

    const auto s_V = mesh_.cellVolumes().field();
    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, mesh_.nCells()), KOKKOS_LAMBDA(const int celli) {
            s_divPhi[celli] *= 1/s_V[celli];
        }
    );
}
void GaussGreenDivKernel::operator()(const CPUExecutor& exec, fvccVolField<scalar>& divPhi, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& phi)
{
    using executor = typename CPUExecutor::exec;
    fvccSurfaceField<NeoFOAM::scalar> phif(exec, mesh_, NeoFOAM::createCalculatedBCs<scalar>(mesh_));
    const auto s_faceCells = mesh_.boundaryMesh().faceCells().field();
    surfaceInterpolation_.interpolate(phif,faceFlux,phi);

    auto s_divPhi = divPhi.internalField().field();
    
    const auto s_phif = phif.internalField().field();
    const auto s_owner = mesh_.faceOwner().field();
    const auto s_neighbour = mesh_.faceNeighbour().field();
    const auto s_faceFlux = faceFlux.internalField().field();
    size_t nInternalFaces = mesh_.nInternalFaces();

    for (int i = 0; i < nInternalFaces; i++)
    {
        NeoFOAM::scalar Flux = s_faceFlux[i] * s_phif[i];
        s_divPhi[s_owner[i]] += Flux;
        s_divPhi[s_neighbour[i]] -= Flux;
    }

    for (int i = nInternalFaces; i < s_phif.size(); i++)
    {
        int32_t own = s_faceCells[i-nInternalFaces];
        NeoFOAM::scalar value_own = s_faceFlux[i] * s_phif[i];
        s_divPhi[own] += value_own;
    }

    const auto s_V = mesh_.cellVolumes().field();
    for (int celli = 0; celli < mesh_.nCells(); celli++)
    {
        s_divPhi[celli] *= 1/s_V[celli];
    }
}


gaussGreenDiv::gaussGreenDiv(const executor& exec, const unstructuredMesh& mesh, const surfaceInterpolation& surfInterp)
    : mesh_(mesh),
      surfaceInterpolation_(surfInterp) {

      };

void gaussGreenDiv::div(fvccVolField<scalar>& divPhi, const fvccSurfaceField<scalar>& faceFlux, fvccVolField<scalar>& phi)
{
    GaussGreenDivKernel kernel_(mesh_, surfaceInterpolation_);
    std::visit([&](const auto& exec)
               { kernel_(exec,divPhi,faceFlux,phi); },
               divPhi.exec());
};

};