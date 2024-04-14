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

void GaussGreenDivKernel::operator()(const GPUExecutor& exec, fvccVolField<scalar>& gradPhi, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& phi)
{
    using executor = typename GPUExecutor::exec;
    fvccSurfaceField<NeoFOAM::scalar> phif(exec, mesh_, NeoFOAM::createCalculatedBCs<scalar>(mesh_));
    surfaceInterpolation_.interpolate(phif,phi);
    auto s_phif = phif.internalField().field();
    auto s_gradPhi = gradPhi.internalField().field();
    auto s_phi = phi.internalField().field();
    auto s_owner = mesh_.faceOwner().field();
    auto s_neighbour = mesh_.faceNeighbour().field();
    auto s_Sf = mesh_.faceAreas().field();
    auto s_V = mesh_.cellVolumes().field();
    auto s_faceFlux = faceFlux.internalField().field();

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, mesh_.nInternalFaces()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_owner[i];
            int32_t nei = s_neighbour[i];
            NeoFOAM::scalar phif = s_phif[i];
            NeoFOAM::scalar value_own = s_faceFlux[i] * (phif / s_V[own]);
            NeoFOAM::scalar value_nei = s_faceFlux[i] * (phif / s_V[nei]);
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
            Kokkos::atomic_sub(&s_gradPhi[nei], value_nei);
        }
    );

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(mesh_.nInternalFaces(), s_phif.size()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_owner[i];
            NeoFOAM::scalar value_own = s_faceFlux[i] * (s_phif[i] / s_V[own]);
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
        }
    );
}

void GaussGreenDivKernel::operator()(const OMPExecutor& exec, fvccVolField<scalar>& gradPhi, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& phi)
{
    using executor = typename GPUExecutor::exec;
    fvccSurfaceField<NeoFOAM::scalar> phif(exec, mesh_, NeoFOAM::createCalculatedBCs<scalar>(mesh_));
    surfaceInterpolation_.interpolate(phif,phi);
    auto s_phif = phif.internalField().field();
    auto s_gradPhi = gradPhi.internalField().field();
    auto s_phi = phi.internalField().field();
    auto s_owner = mesh_.faceOwner().field();
    auto s_neighbour = mesh_.faceNeighbour().field();
    auto s_Sf = mesh_.faceAreas().field();
    auto s_V = mesh_.cellVolumes().field();
    auto s_faceFlux = faceFlux.internalField().field();

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, mesh_.nInternalFaces()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_owner[i];
            int32_t nei = s_neighbour[i];
            NeoFOAM::scalar phif = s_phif[i];
            NeoFOAM::scalar value_own = s_faceFlux[i] * (phif / s_V[own]);
            NeoFOAM::scalar value_nei = s_faceFlux[i] * (phif / s_V[nei]);
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
            Kokkos::atomic_sub(&s_gradPhi[nei], value_nei);
        }
    );

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(mesh_.nInternalFaces(), s_phif.size()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_owner[i];
            NeoFOAM::scalar value_own = s_faceFlux[i] * (s_phif[i] / s_V[own]);
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
        }
    );
}
void GaussGreenDivKernel::operator()(const CPUExecutor& exec, fvccVolField<scalar>& gradPhi, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& phi)
{
    using executor = typename CPUExecutor::exec;
    fvccSurfaceField<NeoFOAM::scalar> phif(exec, mesh_, NeoFOAM::createCalculatedBCs<scalar>(mesh_));
    surfaceInterpolation_.interpolate(phif,phi);
    auto s_phif = phif.internalField().field();
    auto s_gradPhi = gradPhi.internalField().field();
    auto s_phi = phi.internalField().field();
    auto s_owner = mesh_.faceOwner().field();
    auto s_neighbour = mesh_.faceNeighbour().field();
    auto s_Sf = mesh_.faceAreas().field();
    auto s_V = mesh_.cellVolumes().field();
    auto s_faceFlux = faceFlux.internalField().field();

    for (int i = 0; i < mesh_.nInternalFaces(); i++)
    {
        int32_t own = s_owner[i];
        int32_t nei = s_neighbour[i];
        NeoFOAM::scalar phif = s_phif[i];
        NeoFOAM::scalar value_own = s_faceFlux[i] * (phif / s_V[own]);
        NeoFOAM::scalar value_nei = s_faceFlux[i] * (phif / s_V[nei]);
        s_gradPhi[own] += value_own;
        s_gradPhi[nei] -= value_nei;
    }
    for (int i = mesh_.nInternalFaces(); i < s_phif.size(); i++)
    {
        int32_t own = s_owner[i];
        NeoFOAM::scalar value_own = s_faceFlux[i] * (s_phif[i] / s_V[own]);
        s_gradPhi[own] += value_own;
    }
}


gaussGreenDiv::gaussGreenDiv(const executor& exec, const unstructuredMesh& mesh)
    : mesh_(mesh),
      surfaceInterpolation_(exec, mesh, std::make_unique<NeoFOAM::linear>(exec, mesh)) {

      };

// const vectorField& gaussGreenDiv::grad(const scalarField& phi)
// {
//     GaussGreenDivKernel kernel_(mesh_, phi, gradPhi_);
//     std::visit([&](const auto& exec)
//                { kernel_(exec); },
//                gradPhi_.exec());

//     return gradPhi_;
// };


// void gaussGreenDiv::grad(vectorField& gradPhi, const scalarField& phi)
// {
//     GaussGreenDivKernel kernel_(mesh_, phi, gradPhi);
//     std::visit([&](const auto& exec)
//                { kernel_(exec); },
//                gradPhi.exec());
// };

};