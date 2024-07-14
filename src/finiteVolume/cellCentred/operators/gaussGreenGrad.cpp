// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/operators/gaussGreenGrad.hpp"
#include <functional>

namespace NeoFOAM
{

GaussGreenKernel::GaussGreenKernel(
    const UnstructuredMesh& mesh, const SurfaceInterpolation& surfInterp
)
    : mesh_(mesh), surfaceInterpolation_(surfInterp) {};

void GaussGreenKernel::operator()(
    const GPUExecutor& exec,
    fvcc::VolumeField<Vector>& gradPhi,
    const fvcc::VolumeField<scalar>& phi
)
{
    using executor = typename GPUExecutor::exec;
    fvcc::SurfaceField<NeoFOAM::scalar> phif(
        exec, mesh_, NeoFOAM::createCalculatedBCs<scalar>(mesh_)
    );
    const auto s_faceCells = mesh_.boundaryMesh().faceCells().span();
    surfaceInterpolation_.interpolate(phif, phi);

    const auto s_phif = phif.internalField().span();
    const auto s_gradPhi = gradPhi.internalField().span();
    const auto s_owner = mesh_.faceOwner().span();
    const auto s_neighbour = mesh_.faceNeighbour().span();
    const auto s_Sf = mesh_.faceAreas().span();
    size_t nInternalFaces = mesh_.nInternalFaces();

    Kokkos::parallel_for(
        "gaussGreenGrad",
        Kokkos::RangePolicy<executor>(0, nInternalFaces),
        KOKKOS_LAMBDA(const int i) {
            NeoFOAM::Vector Flux = s_Sf[i] * s_phif[i];
            Kokkos::atomic_add(&s_gradPhi[s_owner[i]], Flux);
            Kokkos::atomic_sub(&s_gradPhi[s_neighbour[i]], Flux);
        }
    );

    Kokkos::parallel_for(
        "gaussGreenGrad",
        Kokkos::RangePolicy<executor>(nInternalFaces, s_phif.size()),
        KOKKOS_LAMBDA(const int i) {
            int32_t own = s_faceCells[i - nInternalFaces];
            NeoFOAM::Vector value_own = s_Sf[i] * s_phif[i];
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
        }
    );

    const auto s_V = mesh_.cellVolumes().span();
    Kokkos::parallel_for(
        "gaussGreenGrad",
        Kokkos::RangePolicy<executor>(0, mesh_.nCells()),
        KOKKOS_LAMBDA(const int celli) { s_gradPhi[celli] *= 1 / s_V[celli]; }
    );
}

void GaussGreenKernel::operator()(
    const OMPExecutor& exec,
    fvcc::VolumeField<Vector>& gradPhi,
    const fvcc::VolumeField<scalar>& phi
)
{
    using executor = typename OMPExecutor::exec;
    fvcc::SurfaceField<NeoFOAM::scalar> phif(
        exec, mesh_, NeoFOAM::createCalculatedBCs<scalar>(mesh_)
    );
    const auto s_faceCells = mesh_.boundaryMesh().faceCells().span();
    surfaceInterpolation_.interpolate(phif, phi);

    const auto s_phif = phif.internalField().span();
    const auto s_gradPhi = gradPhi.internalField().span();
    const auto s_owner = mesh_.faceOwner().span();
    const auto s_neighbour = mesh_.faceNeighbour().span();
    const auto s_Sf = mesh_.faceAreas().span();
    size_t nInternalFaces = mesh_.nInternalFaces();

    Kokkos::parallel_for(
        "gaussGreenGrad",
        Kokkos::RangePolicy<executor>(0, nInternalFaces),
        KOKKOS_LAMBDA(const int i) {
            NeoFOAM::Vector Flux = s_Sf[i] * s_phif[i];
            Kokkos::atomic_add(&s_gradPhi[s_owner[i]], Flux);
            Kokkos::atomic_sub(&s_gradPhi[s_neighbour[i]], Flux);
        }
    );

    Kokkos::parallel_for(
        "gaussGreenGrad",
        Kokkos::RangePolicy<executor>(nInternalFaces, s_phif.size()),
        KOKKOS_LAMBDA(const int i) {
            int32_t own = s_faceCells[i - nInternalFaces];
            NeoFOAM::Vector value_own = s_Sf[i] * s_phif[i];
            Kokkos::atomic_add(&s_gradPhi[own], value_own);
        }
    );

    const auto s_V = mesh_.cellVolumes().span();
    Kokkos::parallel_for(
        "gaussGreenGrad",
        Kokkos::RangePolicy<executor>(0, mesh_.nCells()),
        KOKKOS_LAMBDA(const int celli) { s_gradPhi[celli] *= 1 / s_V[celli]; }
    );
}

void GaussGreenKernel::operator()(
    const CPUExecutor& exec,
    fvcc::VolumeField<Vector>& gradPhi,
    const fvcc::VolumeField<scalar>& phi
)
{
    using executor = typename CPUExecutor::exec;
    fvcc::SurfaceField<NeoFOAM::scalar> phif(
        exec, mesh_, NeoFOAM::createCalculatedBCs<scalar>(mesh_)
    );
    const auto s_faceCells = mesh_.boundaryMesh().faceCells().span();
    const auto s_bSf = mesh_.boundaryMesh().sf().span();
    auto s_gradPhi = gradPhi.internalField().span();

    surfaceInterpolation_.interpolate(phif, phi);
    const auto s_phif = phif.internalField().span();
    const auto s_owner = mesh_.faceOwner().span();
    const auto s_neighbour = mesh_.faceNeighbour().span();
    const auto s_Sf = mesh_.faceAreas().span();
    size_t nInternalFaces = mesh_.nInternalFaces();

    for (int i = 0; i < nInternalFaces; i++)
    {
        NeoFOAM::Vector Flux = s_Sf[i] * s_phif[i];
        s_gradPhi[s_owner[i]] += Flux;
        s_gradPhi[s_neighbour[i]] -= Flux;
    }

    for (int i = nInternalFaces; i < s_phif.size(); i++)
    {
        int32_t own = s_faceCells[i - nInternalFaces];
        NeoFOAM::Vector value_own = s_bSf[i - nInternalFaces] * s_phif[i];
        s_gradPhi[own] += value_own;
    }

    const auto s_V = mesh_.cellVolumes().span();
    for (int celli = 0; celli < mesh_.nCells(); celli++)
    {
        s_gradPhi[celli] *= 1 / s_V[celli];
    }
}


gaussGreenGrad::gaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh)
    : mesh_(mesh),
      surfaceInterpolation_(exec, mesh, std::make_unique<NeoFOAM::Linear>(exec, mesh)) {

      };


void gaussGreenGrad::grad(fvcc::VolumeField<Vector>& gradPhi, const fvcc::VolumeField<scalar>& phi)
{
    GaussGreenKernel kernel_(mesh_, surfaceInterpolation_);
    std::visit([&](const auto& exec) { kernel_(exec, gradPhi, phi); }, gradPhi.exec());
};

// fvcc::VolumeField<Vector> gaussGreenGrad::grad(const fvcc::VolumeField<scalar>& phi)
// {
//     fvcc::VolumeField<Vector> gradPhi(phi.exec(), mesh_, phi);
//     GaussGreenKernel kernel_(mesh_, surfaceInterpolation_);
//     std::visit([&](const auto& exec)
//                { kernel_(exec,gradPhi,phi); },
//                gradPhi.exec());

//     return gradPhi;
// };

} // namespace NeoFOAM
