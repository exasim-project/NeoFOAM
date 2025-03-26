// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

struct SurfaceFluxView
{
    const std::span<const Vector> flux;
    const std::span<const scalar> phif;
    const std::span<const int> owner;
    const std::span<const int> neigh;
    const std::span<const int> cells;
};

void computeGrad(
    const VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp,
    VolumeField<Vector>& gradPhi,
    SurfaceField<scalar>& phif
)
{
    const UnstructuredMesh& mesh = gradPhi.mesh();
    const auto exec = gradPhi.exec();
    auto surfGradPhi = gradPhi.internalField().span();

    surfInterp.interpolate(phi, phif);
    size_t nInternalFaces = mesh.nInternalFaces();

    const SurfaceFluxView surfView {
        mesh.faceAreas().span(),
        phif.internalField().span(),
        mesh.faceOwner().span(),
        mesh.faceNeighbour().span(),
        mesh.boundaryMesh().faceCells().span()
    };

    std::visit(
        [&](const auto& e)
        {
            AtomicAdd add {e};
            AtomicSub sub {e};
            parallelFor(
                e,
                {0, nInternalFaces},
                KOKKOS_LAMBDA(const size_t i) {
                    Vector flux = surfView.flux[i] * surfView.phif[i];
                    add(surfGradPhi[static_cast<size_t>(surfView.owner[i])], flux);
                    sub(surfGradPhi[static_cast<size_t>(surfView.neigh[i])], flux);
                },
                {nInternalFaces, phif.size()},
                KOKKOS_LAMBDA(const size_t i) {
                    auto own = static_cast<size_t>(surfView.cells[i - nInternalFaces]);
                    Vector valueOwn = surfView.flux[i] * surfView.phif[i];
                    add(surfGradPhi[own], valueOwn);
                },
                "sumFluxes"
            );
        },
        exec
    );

    const auto surfV = mesh.cellVolumes().span();

    parallelFor(
        exec,
        {0, mesh.nCells()},
        KOKKOS_LAMBDA(const size_t celli) { surfGradPhi[celli] *= 1 / surfV[celli]; }
    );
}

void computeGrad(
    const VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp,
    VolumeField<Vector>& gradPhi
)
{
    const auto exec = gradPhi.exec();
    const UnstructuredMesh& mesh = gradPhi.mesh();
    SurfaceField<scalar> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
    );

    computeGrad(phi, surfInterp, gradPhi, phif);
}

GaussGreenGrad::GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh)
    : mesh_(mesh),
      surfaceInterpolation_(exec, mesh, std::make_unique<Linear>(exec, mesh, Dictionary())) {};


void GaussGreenGrad::grad(const VolumeField<scalar>& phi, VolumeField<Vector>& gradPhi)
{
    computeGrad(phi, surfaceInterpolation_, gradPhi);
};

} // namespace NeoFOAM
