// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/grad/gaussGreenGrad.hpp"
#include <functional>

namespace NeoFOAM
{

GaussGreenKernel::GaussGreenKernel(const unstructuredMesh& mesh, const scalarField& phi, vectorField& gradPhi)
    : mesh_(mesh), phi_(phi), gradPhi_(gradPhi)
     {
        NeoFOAM::fill(gradPhi_, NeoFOAM::vector(0.0, 0.0, 0.0));
     };

void GaussGreenKernel::operator()(const GPUExecutor& exec)
{
    using executor = typename GPUExecutor::exec;
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();
    const NeoFOAM::vectorField& Sf = mesh_.faceAreas();
    const NeoFOAM::scalarField& V = mesh_.cellVolumes();
    auto s_gradPhi = gradPhi_.field();
    auto s_phi = phi_.field();
    auto s_owner = owner.field();
    auto s_neighbour = neighbour.field();
    auto s_Sf = Sf.field();
    auto s_V = V.field();

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, mesh_.nInternalFaces()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_owner[i];
            int32_t nei = s_neighbour[i];
            NeoFOAM::scalar phif = 0.5 * (s_phi[nei] + s_phi[own]);
            NeoFOAM::vector value_own = s_Sf[i] * (phif / s_V[own]);
            NeoFOAM::vector value_nei = s_Sf[i] * (phif / s_V[nei]);
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
            Kokkos::atomic_sub(&s_gradPhi[nei], value_nei);
        }
    );
}

void GaussGreenKernel::operator()(const OMPExecutor& exec)
{
    using executor = typename OMPExecutor::exec;
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();
    const NeoFOAM::vectorField& Sf = mesh_.faceAreas();
    const NeoFOAM::scalarField& V = mesh_.cellVolumes();
    auto s_gradPhi = gradPhi_.field();
    auto s_phi = phi_.field();
    auto s_owner = owner.field();
    auto s_neighbour = neighbour.field();
    auto s_Sf = Sf.field();
    auto s_V = V.field();

    Kokkos::parallel_for(
        "gaussGreenGrad", Kokkos::RangePolicy<executor>(0, mesh_.nInternalFaces()), KOKKOS_LAMBDA(const int i) {
            int32_t own = s_owner[i];
            int32_t nei = s_neighbour[i];
            NeoFOAM::scalar phif = 0.5 * (s_phi[nei] + s_phi[own]);
            NeoFOAM::vector value_own = s_Sf[i] * (phif / s_V[own]);
            NeoFOAM::vector value_nei = s_Sf[i] * (phif / s_V[nei]);
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
            Kokkos::atomic_sub(&s_gradPhi[nei], value_nei);
        }
    );

}

void GaussGreenKernel::operator()(const CPUExecutor& exec)
{
    using executor = typename CPUExecutor::exec;
    const NeoFOAM::labelField& owner = mesh_.faceOwner();
    const NeoFOAM::labelField& neighbour = mesh_.faceNeighbour();
    const NeoFOAM::vectorField& Sf = mesh_.faceAreas();
    const NeoFOAM::scalarField& V = mesh_.cellVolumes();
    auto s_gradPhi = gradPhi_.field();
    auto s_phi = phi_.field();
    auto s_owner = owner.field();
    auto s_neighbour = neighbour.field();
    auto s_Sf = Sf.field();
    auto s_V = V.field();

    for (int i = 0; i < mesh_.nInternalFaces(); i++)
    {
            int32_t own = s_owner[i];
            int32_t nei = s_neighbour[i];
            NeoFOAM::scalar phif = 0.5 * (s_phi[nei] + s_phi[own]);
            NeoFOAM::vector value_own = (s_Sf[i] * (phif / s_V[own]));
            NeoFOAM::vector value_nei = (s_Sf[i] * (phif / s_V[nei]));
            s_gradPhi[own] += value_own;
            s_gradPhi[nei] -= value_nei;
    }
}


gaussGreenGrad::gaussGreenGrad(const executor& exec, const unstructuredMesh& mesh)
    : mesh_(mesh), gradPhi_(exec, mesh.nCells()) {};

const vectorField& gaussGreenGrad::grad(const scalarField& phi)
{
    GaussGreenKernel kernel_(mesh_, phi, gradPhi_);
    std::visit([&](const auto& exec)
               { kernel_(exec); },
               gradPhi_.exec());

    return gradPhi_;
};


void gaussGreenGrad::grad(vectorField& gradPhi, const scalarField& phi)
{
    GaussGreenKernel kernel_(mesh_, phi, gradPhi);
    std::visit([&](const auto& exec)
               { kernel_(exec); },
               gradPhi.exec());
};

};