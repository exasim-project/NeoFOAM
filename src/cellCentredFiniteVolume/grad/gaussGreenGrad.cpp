// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/grad/gaussGreenGrad.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryFieldSelector.hpp"
#include <functional>

namespace NeoFOAM
{

GaussGreenKernel::GaussGreenKernel(const unstructuredMesh& mesh, const surfaceInterpolation& surfInterp)
    : mesh_(mesh),
      surfaceInterpolation_(surfInterp)
{
};

void GaussGreenKernel::operator()(const GPUExecutor& exec, fvccVolField<Vector>& gradPhi, const fvccVolField<scalar>& phi)
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

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, mesh_.nInternalFaces()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_owner[i];
            int32_t nei = s_neighbour[i];
            NeoFOAM::scalar phif = 0.5 * (s_phi[nei] + s_phi[own]);
            NeoFOAM::Vector value_own = s_Sf[i] * (phif / s_V[own]);
            NeoFOAM::Vector value_nei = s_Sf[i] * (phif / s_V[nei]);
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
            Kokkos::atomic_sub(&s_gradPhi[nei], value_nei);
        }
    );

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(mesh_.nInternalFaces(), s_phif.size()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_owner[i];
            NeoFOAM::Vector value_own = s_Sf[i] * (s_phif[i] / s_V[own]);
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
        }
    );
}

void GaussGreenKernel::operator()(const OMPExecutor& exec, fvccVolField<Vector>& gradPhi, const fvccVolField<scalar>& phi)
{
    using executor = typename OMPExecutor::exec;
    fvccSurfaceField<NeoFOAM::scalar> phif(exec, mesh_, NeoFOAM::createCalculatedBCs<scalar>(mesh_));
    surfaceInterpolation_.interpolate(phif,phi);
    auto s_phif = phif.internalField().field();
    auto s_gradPhi = gradPhi.internalField().field();
    auto s_phi = phi.internalField().field();
    auto s_owner = mesh_.faceOwner().field();
    auto s_neighbour = mesh_.faceNeighbour().field();
    auto s_Sf = mesh_.faceAreas().field();
    auto s_V = mesh_.cellVolumes().field();

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, mesh_.nInternalFaces()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_owner[i];
            int32_t nei = s_neighbour[i];
            NeoFOAM::scalar phif = 0.5 * (s_phi[nei] + s_phi[own]);
            NeoFOAM::Vector value_own = s_Sf[i] * (phif / s_V[own]);
            NeoFOAM::Vector value_nei = s_Sf[i] * (phif / s_V[nei]);
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
            Kokkos::atomic_sub(&s_gradPhi[nei], value_nei);
        }
    );

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(mesh_.nInternalFaces(), s_phif.size()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_owner[i];
            NeoFOAM::Vector value_own = s_Sf[i] * (s_phif[i] / s_V[own]);
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
        }
    );
}

void GaussGreenKernel::operator()(const CPUExecutor& exec, fvccVolField<Vector>& gradPhi, const fvccVolField<scalar>& phi)
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

    for (int i = 0; i < mesh_.nInternalFaces(); i++)
    {
        int32_t own = s_owner[i];
        int32_t nei = s_neighbour[i];
        NeoFOAM::scalar phif = s_phif[i];
        NeoFOAM::Vector value_own = s_Sf[i] * (phif / s_V[own]);
        NeoFOAM::Vector value_nei = s_Sf[i] * (phif / s_V[nei]);
        s_gradPhi[own] += value_own;
        s_gradPhi[nei] -= value_nei;
    }
    for (int i = mesh_.nInternalFaces(); i < s_phif.size(); i++)
    {
        int32_t own = s_owner[i];
        NeoFOAM::Vector value_own = s_Sf[i] * (s_phif[i] / s_V[own]);
        s_gradPhi[own] += value_own;
    }
}


gaussGreenGrad::gaussGreenGrad(const executor& exec, const unstructuredMesh& mesh)
    : mesh_(mesh),
    surfaceInterpolation_(exec, mesh, std::make_unique<NeoFOAM::linear>(exec, mesh))
{

};



void gaussGreenGrad::grad(fvccVolField<Vector>& gradPhi, const fvccVolField<scalar>& phi)
{
    GaussGreenKernel kernel_(mesh_, surfaceInterpolation_);
    std::visit([&](const auto& exec)
               { kernel_(exec,gradPhi,phi); },
               gradPhi.exec());
};

// fvccVolField<Vector> gaussGreenGrad::grad(const fvccVolField<scalar>& phi)
// {
//     fvccVolField<Vector> gradPhi(phi.exec(), mesh_, phi);
//     GaussGreenKernel kernel_(mesh_, surfaceInterpolation_);
//     std::visit([&](const auto& exec)
//                { kernel_(exec,gradPhi,phi); },
//                gradPhi.exec());

//     return gradPhi;
// };

} // namespace NeoFOAM

