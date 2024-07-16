// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <functional>

#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/mesh/stencil/fvccGeometryScheme.hpp"

namespace NeoFOAM
{


void computeDiv(
    fvcc::VolumeField<scalar>& divPhi,
    const fvcc::SurfaceField<scalar>& faceFlux,
    const fvcc::VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp
)
{

    const UnstructuredMesh& mesh = divPhi.mesh();
    const auto exec = divPhi.exec();
    fvcc::SurfaceField<NeoFOAM::scalar> phif(
        exec, mesh, NeoFOAM::createCalculatedBCs<scalar>(mesh)
    );
    const auto s_faceCells = mesh.boundaryMesh().faceCells().span();
    surfInterp.interpolate(phif, faceFlux, phi);

    auto s_divPhi = divPhi.internalField().span();

    const auto s_phif = phif.internalField().span();
    const auto s_owner = mesh.faceOwner().span();
    const auto s_neighbour = mesh.faceNeighbour().span();
    const auto s_faceFlux = faceFlux.internalField().span();
    size_t nInternalFaces = mesh.nInternalFaces();
    const auto s_V = mesh.cellVolumes().span();


    // check if the executor is GPU
    if (std::holds_alternative<CPUExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            NeoFOAM::scalar Flux = s_faceFlux[i] * s_phif[i];
            s_divPhi[s_owner[i]] += Flux;
            s_divPhi[s_neighbour[i]] -= Flux;
        }

        for (size_t i = nInternalFaces; i < s_phif.size(); i++)
        {
            int32_t own = s_faceCells[i - nInternalFaces];
            NeoFOAM::scalar value_own = s_faceFlux[i] * s_phif[i];
            s_divPhi[own] += value_own;
        }


        for (size_t celli = 0; celli < mesh.nCells(); celli++)
        {
            s_divPhi[celli] *= 1 / s_V[celli];
        }
    }
    else
    {
        NeoFOAM::parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                NeoFOAM::scalar Flux = s_faceFlux[i] * s_phif[i];
                Kokkos::atomic_add(&s_divPhi[s_owner[i]], Flux);
                Kokkos::atomic_sub(&s_divPhi[s_neighbour[i]], Flux);
            }
        );

        NeoFOAM::parallelFor(
            exec,
            {nInternalFaces, s_phif.size()},
            KOKKOS_LAMBDA(const size_t i) {
                int32_t own = s_faceCells[i - nInternalFaces];
                NeoFOAM::scalar value_own = s_faceFlux[i] * s_phif[i];
                Kokkos::atomic_add(&s_divPhi[own], value_own);
            }
        );

        NeoFOAM::parallelFor(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli) { s_divPhi[celli] *= 1 / s_V[celli]; }
        );
    }
}

GaussGreenDiv::GaussGreenDiv(
    const Executor& exec, const UnstructuredMesh& mesh, const SurfaceInterpolation& surfInterp
)
    : mesh_(mesh), surfaceInterpolation_(surfInterp) {

                   };

void GaussGreenDiv::div(
    fvcc::VolumeField<scalar>& divPhi,
    const fvcc::SurfaceField<scalar>& faceFlux,
    fvcc::VolumeField<scalar>& phi
)
{
    computeDiv(divPhi, faceFlux, phi, surfaceInterpolation_);
};

};
