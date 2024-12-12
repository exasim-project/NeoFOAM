// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

void computeGrad(
    const VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp,
    VolumeField<Vector>& gradPhi
)
{
    const UnstructuredMesh& mesh = gradPhi.mesh();
    const auto exec = gradPhi.exec();
    SurfaceField<scalar> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
    );
    const auto surfFaceCells = mesh.boundaryMesh().faceCells().span();
    const auto sBSf = mesh.boundaryMesh().sf().span();
    auto surfGradPhi = gradPhi.internalField().span();

    surfInterp.interpolate(phi, phif);
    const auto surfPhif = phif.internalField().span();
    const auto surfOwner = mesh.faceOwner().span();
    const auto surfNeighbour = mesh.faceNeighbour().span();
    const auto sSf = mesh.faceAreas().span();
    size_t nInternalFaces = mesh.nInternalFaces();

    const auto surfV = mesh.cellVolumes().span();

    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            Vector flux = sSf[i] * surfPhif[i];
            surfGradPhi[static_cast<size_t>(surfOwner[i])] += flux;
            surfGradPhi[static_cast<size_t>(surfNeighbour[i])] -= flux;
        }

        for (size_t i = nInternalFaces; i < surfPhif.size(); i++)
        {
            size_t own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
            Vector valueOwn = sBSf[i - nInternalFaces] * surfPhif[i];
            surfGradPhi[own] += valueOwn;
        }

        for (size_t celli = 0; celli < mesh.nCells(); celli++)
        {
            surfGradPhi[celli] *= 1 / surfV[celli];
        }
    }
    else
    {
        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                Vector flux = sSf[i] * surfPhif[i];
                Kokkos::atomic_add(&surfGradPhi[static_cast<size_t>(surfOwner[i])], flux);
                Kokkos::atomic_sub(&surfGradPhi[static_cast<size_t>(surfNeighbour[i])], flux);
            }
        );

        parallelFor(
            exec,
            {nInternalFaces, surfPhif.size()},
            KOKKOS_LAMBDA(const size_t i) {
                size_t own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
                Vector valueOwn = sSf[i] * surfPhif[i];
                Kokkos::atomic_add(&surfGradPhi[own], valueOwn);
            }
        );

        parallelFor(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli) { surfGradPhi[celli] *= 1 / surfV[celli]; }
        );
    }
}

GaussGreenGrad::GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh)
    : mesh_(mesh),
      surfaceInterpolation_(exec, mesh, std::make_unique<Linear>(exec, mesh, Dictionary())) {};


void GaussGreenGrad::grad(const VolumeField<scalar>& phi, VolumeField<Vector>& gradPhi)
{
    computeGrad(phi, surfaceInterpolation_, gradPhi);
};

} // namespace NeoFOAM
