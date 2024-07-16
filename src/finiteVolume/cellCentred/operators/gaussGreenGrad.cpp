// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <functional>

#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM
{

void computeGrad(
    fvcc::VolumeField<Vector>& gradPhi,
    const fvcc::VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp
)
{

    const UnstructuredMesh& mesh = gradPhi.mesh();
    const auto exec = gradPhi.exec();
    fvcc::SurfaceField<NeoFOAM::scalar> phif(
        exec, mesh, NeoFOAM::createCalculatedBCs<scalar>(mesh)
    );
    const auto s_faceCells = mesh.boundaryMesh().faceCells().span();
    const auto s_bSf = mesh.boundaryMesh().sf().span();
    auto s_gradPhi = gradPhi.internalField().span();

    surfInterp.interpolate(phif, phi);
    const auto s_phif = phif.internalField().span();
    const auto s_owner = mesh.faceOwner().span();
    const auto s_neighbour = mesh.faceNeighbour().span();
    const auto s_Sf = mesh.faceAreas().span();
    size_t nInternalFaces = mesh.nInternalFaces();

    const auto s_V = mesh.cellVolumes().span();

    if (std::holds_alternative<CPUExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            NeoFOAM::Vector Flux = s_Sf[i] * s_phif[i];
            s_gradPhi[s_owner[i]] += Flux;
            s_gradPhi[s_neighbour[i]] -= Flux;
        }

        for (size_t i = nInternalFaces; i < s_phif.size(); i++)
        {
            int32_t own = s_faceCells[i - nInternalFaces];
            NeoFOAM::Vector value_own = s_bSf[i - nInternalFaces] * s_phif[i];
            s_gradPhi[own] += value_own;
        }

        for (size_t celli = 0; celli < mesh.nCells(); celli++)
        {
            s_gradPhi[celli] *= 1 / s_V[celli];
        }
    }
    else
    {
        NeoFOAM::parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                NeoFOAM::Vector Flux = s_Sf[i] * s_phif[i];
                Kokkos::atomic_add(&s_gradPhi[s_owner[i]], Flux);
                Kokkos::atomic_sub(&s_gradPhi[s_neighbour[i]], Flux);
            }
        );

        NeoFOAM::parallelFor(
            exec,
            {nInternalFaces, s_phif.size()},
            KOKKOS_LAMBDA(const size_t i) {
                size_t own = s_faceCells[i - nInternalFaces];
                NeoFOAM::Vector value_own = s_Sf[i] * s_phif[i];
                Kokkos::atomic_add(&s_gradPhi[own], value_own);
            }
        );

        NeoFOAM::parallelFor(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli) { s_gradPhi[celli] *= 1 / s_V[celli]; }
        );
    }
}

gaussGreenGrad::gaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh)
    : mesh_(mesh),
      surfaceInterpolation_(exec, mesh, std::make_unique<NeoFOAM::Linear>(exec, mesh)) {

      };


void gaussGreenGrad::grad(fvcc::VolumeField<Vector>& gradPhi, const fvcc::VolumeField<scalar>& phi)
{
    computeGrad(gradPhi, phi, surfaceInterpolation_);
};

} // namespace NeoFOAM
