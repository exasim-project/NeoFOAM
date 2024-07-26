// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <functional>

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/stencil/geometryScheme.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


void computeDiv(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp,
    VolumeField<scalar>& divPhi
)
{
    const UnstructuredMesh& mesh = divPhi.mesh();
    const auto exec = divPhi.exec();
    SurfaceField<scalar> phif(exec, mesh, createCalculatedBCs<scalar>(mesh));
    const auto surfFaceCells = mesh.boundaryMesh().faceCells().span();
    surfInterp.interpolate(phif, faceFlux, phi);

    auto surfDivPhi = divPhi.internalField().span();

    const auto surfPhif = phif.internalField().span();
    const auto surfOwner = mesh.faceOwner().span();
    const auto surfNeighbour = mesh.faceNeighbour().span();
    const auto surfFaceFlux = faceFlux.internalField().span();
    size_t nInternalFaces = mesh.nInternalFaces();
    const auto surfV = mesh.cellVolumes().span();


    // check if the executor is GPU
    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            scalar flux = surfFaceFlux[i] * surfPhif[i];
            surfDivPhi[surfOwner[i]] += flux;
            surfDivPhi[surfNeighbour[i]] -= flux;
        }

        for (size_t i = nInternalFaces; i < surfPhif.size(); i++)
        {
            int32_t own = surfFaceCells[i - nInternalFaces];
            scalar valueOwn = surfFaceFlux[i] * surfPhif[i];
            surfDivPhi[own] += valueOwn;
        }


        for (size_t celli = 0; celli < mesh.nCells(); celli++)
        {
            surfDivPhi[celli] *= 1 / surfV[celli];
        }
    }
    else
    {
        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                scalar flux = surfFaceFlux[i] * surfPhif[i];
                Kokkos::atomic_add(&surfDivPhi[surfOwner[i]], flux);
                Kokkos::atomic_sub(&surfDivPhi[surfNeighbour[i]], flux);
            }
        );

        parallelFor(
            exec,
            {nInternalFaces, surfPhif.size()},
            KOKKOS_LAMBDA(const size_t i) {
                int32_t own = surfFaceCells[i - nInternalFaces];
                scalar valueOwn = surfFaceFlux[i] * surfPhif[i];
                Kokkos::atomic_add(&surfDivPhi[own], valueOwn);
            }
        );

        parallelFor(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli) { surfDivPhi[celli] *= 1 / surfV[celli]; }
        );
    }
}

void computeDiv(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp,
    Field<scalar>& divPhi
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    SurfaceField<scalar> phif(exec, mesh, createCalculatedBCs<scalar>(mesh));
    const auto surfFaceCells = mesh.boundaryMesh().faceCells().span();
    surfInterp.interpolate(phif, faceFlux, phi);

    auto surfDivPhi = divPhi.span();

    const auto surfPhif = phif.internalField().span();
    const auto surfOwner = mesh.faceOwner().span();
    const auto surfNeighbour = mesh.faceNeighbour().span();
    const auto surfFaceFlux = faceFlux.internalField().span();
    size_t nInternalFaces = mesh.nInternalFaces();
    const auto surfV = mesh.cellVolumes().span();


    // check if the executor is GPU
    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            scalar flux = surfFaceFlux[i] * surfPhif[i];
            surfDivPhi[surfOwner[i]] += flux;
            surfDivPhi[surfNeighbour[i]] -= flux;
        }

        for (size_t i = nInternalFaces; i < surfPhif.size(); i++)
        {
            int32_t own = surfFaceCells[i - nInternalFaces];
            scalar valueOwn = surfFaceFlux[i] * surfPhif[i];
            surfDivPhi[own] += valueOwn;
        }


        for (size_t celli = 0; celli < mesh.nCells(); celli++)
        {
            surfDivPhi[celli] *= 1 / surfV[celli];
        }
    }
    else
    {
        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                scalar flux = surfFaceFlux[i] * surfPhif[i];
                Kokkos::atomic_add(&surfDivPhi[surfOwner[i]], flux);
                Kokkos::atomic_sub(&surfDivPhi[surfNeighbour[i]], flux);
            }
        );

        parallelFor(
            exec,
            {nInternalFaces, surfPhif.size()},
            KOKKOS_LAMBDA(const size_t i) {
                int32_t own = surfFaceCells[i - nInternalFaces];
                scalar valueOwn = surfFaceFlux[i] * surfPhif[i];
                Kokkos::atomic_add(&surfDivPhi[own], valueOwn);
            }
        );

        parallelFor(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli) { surfDivPhi[celli] *= 1 / surfV[celli]; }
        );
    }
}

GaussGreenDiv::GaussGreenDiv(
    const Executor& exec, const UnstructuredMesh& mesh, const SurfaceInterpolation& surfInterp
)
    : mesh_(mesh), surfaceInterpolation_(surfInterp) {};

void GaussGreenDiv::div(
    VolumeField<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
)
{
    computeDiv(faceFlux, phi, surfaceInterpolation_, divPhi);
};

void GaussGreenDiv::div(
    Field<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
)
{
    computeDiv(faceFlux, phi, surfaceInterpolation_, divPhi);
};

};
