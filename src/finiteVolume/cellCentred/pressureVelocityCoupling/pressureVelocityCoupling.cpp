// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors


#include "NeoFOAM/finiteVolume/cellCentred/pressureVelocityCoupling/pressureVelocityCoupling.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "Kokkos_Core.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

std::tuple<VolumeField<scalar>, VolumeField<Vector>>
discreteMomentumFields(const Expression<Vector>& expr)
{
    const VolumeField<Vector>& U = expr.getField();
    const UnstructuredMesh& mesh = U.mesh();
    const SparsityPattern& sparsityPattern = expr.sparsityPattern();
    const auto& ls = expr.linearSystem();
    const auto vol = mesh.cellVolumes().span();
    const auto values = ls.matrix().values();
    const auto rhs = ls.rhs().span();
    const auto diagOffset = sparsityPattern.diagOffset().span();
    const auto rowPtrs = ls.matrix().rowPtrs();

    auto rABCs = createCalculatedBCs<VolumeBoundary<scalar>>(mesh);
    VolumeField<scalar> rAU = VolumeField<scalar>(expr.exec(), "rAU", mesh, rABCs);

    rAU.internalField().apply(KOKKOS_LAMBDA(const size_t celli) {
        auto diagOffsetCelli = diagOffset[celli];
        // all the diagonal coefficients are the same
        return vol[celli] / (values[rowPtrs[celli] + diagOffsetCelli][0]);
    });

    auto OffDiagonalSourceBCs = createCalculatedBCs<VolumeBoundary<Vector>>(mesh);
    VolumeField<Vector> HbyA = VolumeField<Vector>(expr.exec(), "HbyA", mesh, OffDiagonalSourceBCs);
    fill(HbyA.internalField(), zero<Vector>());
    const std::size_t nInternalFaces = mesh.nInternalFaces();

    const auto exec = U.exec();
    const auto [owner, neighbour, surfFaceCells, ownOffs, neiOffs, internalU, internalRAU] = spans(
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.boundaryMesh().faceCells(),
        sparsityPattern.ownerOffset(),
        sparsityPattern.neighbourOffset(),
        U.internalField(),
        rAU.internalField()
    );

    auto internalHbyA = HbyA.internalField().span();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            std::size_t own = static_cast<std::size_t>(owner[facei]);
            std::size_t nei = static_cast<std::size_t>(neighbour[facei]);

            std::size_t rowNeiStart = rowPtrs[nei];
            std::size_t rowOwnStart = rowPtrs[own];

            auto Upper = values[rowNeiStart + neiOffs[facei]];
            auto Lower = values[rowOwnStart + ownOffs[facei]];
            Kokkos::atomic_add(&internalHbyA[nei], Lower[0] * internalU[own]);
            Kokkos::atomic_add(&internalHbyA[own], Upper[0] * internalU[nei]);
        }
    );

    parallelFor(
        exec,
        {0, internalHbyA.size()},
        KOKKOS_LAMBDA(const size_t celli) {
            internalHbyA[celli] += rhs[celli];
            internalHbyA[celli] *= internalRAU[celli] / vol[celli];
        }
    );

    return {rAU, HbyA};
}


void updateFaceVelocity(
    SurfaceField<scalar> phi,
    const SurfaceField<scalar> predictedPhi,
    const Expression<scalar>& expr
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const VolumeField<scalar>& p = expr.getField();
    const SparsityPattern sparsityPattern = expr.sparsityPattern();
    const std::size_t nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();
    const auto [owner, neighbour, surfFaceCells, ownOffs, neiOffs, internalPsi] = spans(
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.boundaryMesh().faceCells(),
        sparsityPattern.ownerOffset(),
        sparsityPattern.neighbourOffset(),
        phi.internalField()
    );

    const auto ls = expr.linearSystem();

    const auto rowPtrs = ls.matrix().rowPtrs();
    const auto colIdxs = ls.matrix().colIdxs();
    auto values = ls.matrix().values();
    auto rhs = ls.rhs().span();

    auto [iPhi, iPredPhi] = spans(phi.internalField(), predictedPhi.internalField());

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            std::size_t own = static_cast<std::size_t>(owner[facei]);
            std::size_t nei = static_cast<std::size_t>(neighbour[facei]);

            std::size_t rowNeiStart = rowPtrs[nei];
            std::size_t rowOwnStart = rowPtrs[own];

            auto Upper = values[rowNeiStart + neiOffs[facei]];
            auto Lower = values[rowOwnStart + ownOffs[facei]];
            iPhi[facei] = iPredPhi[facei];
            Kokkos::atomic_add(&iPhi[facei], Upper * internalPsi[nei] - Lower * internalPsi[own]);
        }
    );
}

void updateVelocity(
    VolumeField<Vector>& U,
    const VolumeField<Vector>& HbyA,
    VolumeField<scalar>& rAU,
    VolumeField<scalar>& p
)
{
    VolumeField<Vector> gradP = GaussGreenGrad(p.exec(), p.mesh()).grad(p);
    auto [iHbyA, iRAU, iGradP] =
        spans(HbyA.internalField(), rAU.internalField(), gradP.internalField());

    U.internalField().apply(KOKKOS_LAMBDA(const std::size_t celli) {
        return iHbyA[celli] - iRAU[celli] * iGradP[celli];
    });
}

}
