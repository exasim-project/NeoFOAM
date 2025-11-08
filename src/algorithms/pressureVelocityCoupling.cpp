// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 FoamAdapter authors

#include "NeoN/NeoN.hpp"

#include "FoamAdapter/algorithms/pressureVelocityCoupling.hpp"
#include "Kokkos_Core.hpp"

namespace la = NeoN::la;

namespace FoamAdapter
{

void constrainHbyA(
    const nnfvcc::VolumeField<Vec3>& u,
    const nnfvcc::VolumeField<scalar>& p,
    nnfvcc::VolumeField<Vec3>& hByA
)
{
    auto hByAin = hByA.internalVector().view();
    auto [hByABcValue, uBcValue] = views(hByA.boundaryData().value(), u.boundaryData().value());
    const auto& uBCs = u.boundaryConditions();

    for (auto patchi = 0; patchi < uBCs.size(); ++patchi)
    {
        bool assignable = uBCs[patchi].attributes().assignable;
        if (!assignable)
        {
            parallelFor(
                hByA.exec(),
                hByA.boundaryData().range(patchi),
                KOKKOS_LAMBDA(const size_t bfacei) { hByABcValue[bfacei] = uBcValue[bfacei]; }
            );
        }
    }
}

nnfvcc::VolumeField<scalar> computeRAU(const PDESolver<Vec3>& expr)
{
    // TODO this assumes an assembled matrix
    // force assembly if not assembled
    const auto& mesh = expr.getField().mesh();
    const auto& sparsityPattern = expr.sparsityPattern();
    const auto& ls = expr.linearSystem();

    const auto [vol, values, diagOffset, rowPtrs] = views(
        mesh.cellVolumes(),
        ls.matrix().values(),
        sparsityPattern.diagOffset(),
        ls.matrix().rowOffs()
    );

    auto rABCs = nnfvcc::createExtrapolatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh);
    auto rAU = nnfvcc::VolumeField<scalar>(expr.exec(), "rAU", mesh, rABCs);

    rAU.internalVector().apply(KOKKOS_LAMBDA(const size_t celli) {
        auto diagOffsetCelli = diagOffset[celli];
        // all the diagonal coefficients are the same
        return vol[celli] / (values[rowPtrs[celli] + diagOffsetCelli][0]);
    });

    return rAU;
}

std::tuple<nnfvcc::VolumeField<scalar>, nnfvcc::VolumeField<Vec3>>
computeRAUandHByA(const PDESolver<Vec3>& expr)
{
    const auto& u = expr.getField();
    const auto& mesh = u.mesh();
    const auto& sparsityPattern = expr.sparsityPattern();
    const auto& ls = expr.linearSystem();

    const auto [vol, values, diagOffset, rowPtrs] = views(
        mesh.cellVolumes(),
        ls.matrix().values(),
        sparsityPattern.diagOffset(),
        ls.matrix().rowOffs()
    );

    auto rAU = computeRAU(expr);
    auto offDiagonalSourceBCs = nnfvcc::createExtrapolatedBCs<nnfvcc::VolumeBoundary<Vec3>>(mesh);
    auto hByA = nnfvcc::VolumeField<Vec3>(expr.exec(), "HbyA", mesh, offDiagonalSourceBCs);
    NeoN::fill(hByA.internalVector(), NeoN::zero<Vec3>());
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = u.exec();

    const auto [owner, neighbour, ownOffs, neiOffs, internalU] = views(
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        sparsityPattern.ownerOffset(),
        sparsityPattern.neighbourOffset(),
        u.internalVector()
    );

    auto internalHbyA = hByA.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            auto own = owner[facei];
            auto nei = neighbour[facei];

            auto rowNeiStart = rowPtrs[nei];
            auto rowOwnStart = rowPtrs[own];

            auto lower = values[rowNeiStart + neiOffs[facei]];
            auto upper = values[rowOwnStart + ownOffs[facei]];

            Kokkos::atomic_sub(&internalHbyA[nei], lower[0] * internalU[own]);
            Kokkos::atomic_sub(&internalHbyA[own], upper[0] * internalU[nei]);
        }
    );

    const auto [rhs, internalRAU] = views(ls.rhs(), rAU.internalVector());
    NeoN::parallelFor(
        exec,
        {0, internalHbyA.size()},
        KOKKOS_LAMBDA(const size_t celli) {
            internalHbyA[celli] += rhs[celli];
            internalHbyA[celli] *= internalRAU[celli] / vol[celli];
        }
    );

    hByA.correctBoundaryConditions();
    rAU.correctBoundaryConditions();

    return {rAU, hByA};
}


void updateFaceVelocity(
    const nnfvcc::SurfaceField<scalar>& predictedPhi,
    const PDESolver<scalar>& expr,
    nnfvcc::SurfaceField<scalar>& phi
)
{
    const auto& mesh = phi.mesh();
    const auto& p = expr.getField();
    const auto sparsityPattern = expr.sparsityPattern();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();
    const auto [owner, neighbour, ownOffs, neiOffs, internalP] = views(
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        sparsityPattern.ownerOffset(),
        sparsityPattern.neighbourOffset(),
        p.internalVector()
    );

    const auto& ls = expr.linearSystem();
    const auto rowPtrs = ls.matrix().rowOffs().view();
    const auto colIdxs = ls.matrix().colIdxs().view();
    auto values = ls.matrix().values().view();
    auto rhs = ls.rhs().view();
    auto [iPhi, iPredPhi] = views(phi.internalVector(), predictedPhi.internalVector());

    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            auto own = static_cast<std::size_t>(owner[facei]);
            auto nei = static_cast<std::size_t>(neighbour[facei]);

            auto rowNeiStart = rowPtrs[nei];
            auto rowOwnStart = rowPtrs[own];

            auto upper = values[rowNeiStart + neiOffs[facei]];
            auto lower = values[rowOwnStart + ownOffs[facei]];

            iPhi[facei] = iPredPhi[facei] - (upper * internalP[nei] - lower * internalP[own]);
        }
    );

    auto [bvalue, bPredValue, faceCells] = views(
        phi.boundaryData().value(),
        predictedPhi.boundaryData().value(),
        mesh.boundaryMesh().faceCells()
    );

    auto& bcCoeffs =
        ls.auxiliaryCoefficients().get<la::BoundaryCoefficients<NeoN::scalar, NeoN::localIdx>>(
            "boundaryCoefficients"
        );

    const auto [mValue, rhsValue] = views(bcCoeffs.matrixValues, bcCoeffs.rhsValues);

    NeoN::parallelFor(
        exec,
        {nInternalFaces, iPhi.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            auto bfacei = facei - nInternalFaces;
            scalar bflux = (rhsValue[bfacei] - mValue[bfacei] * internalP[faceCells[bfacei]]);
            iPhi[facei] = iPredPhi[facei] - bflux;
            bvalue[bfacei] = bPredValue[bfacei] - bflux;
        }
    );
}

void updateVelocity(
    const nnfvcc::VolumeField<Vec3>& hByA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& p,
    nnfvcc::VolumeField<Vec3>& u
)
{
    auto gradP = nnfvcc::GaussGreenGrad(p.exec(), p.mesh()).grad(p);
    auto [iHbyA, iRAU, iGradP] =
        views(hByA.internalVector(), rAU.internalVector(), gradP.internalVector());

    u.internalVector().apply(KOKKOS_LAMBDA(const std::size_t celli) {
        return iHbyA[celli] - iRAU[celli] * iGradP[celli];
    });
}

nnfvcc::SurfaceField<scalar> flux(const nnfvcc::VolumeField<Vec3>& volField)
{
    const auto exec = volField.exec();

    const auto& mesh = volField.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    NeoN::Input input = NeoN::TokenList({std::string("linear")});
    auto linear = nnfvcc::SurfaceInterpolation<Vec3>(exec, mesh, input);
    const auto weight = linear.weight(volField);

    auto surfaceBCs = nnfvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh);
    auto faceFlux = nnfvcc::SurfaceField<scalar>(exec, "out", mesh, surfaceBCs);

    NeoN::fill(faceFlux.internalVector(), NeoN::zero<scalar>());
    NeoN::fill(faceFlux.boundaryData().value(), NeoN::zero<scalar>());
    const auto [owner, neighbour, weightIn, faceAreas, volFieldIn, volFieldBc, bSf] = views(
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        weight.internalVector(),
        mesh.faceAreas(),
        volField.internalVector(),
        volField.boundaryData().value(),
        mesh.boundaryMesh().sf()
    );

    auto [faceFluxIn, bvalue] = views(faceFlux.internalVector(), faceFlux.boundaryData().value());

    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            auto own = static_cast<std::size_t>(owner[facei]);
            auto nei = static_cast<std::size_t>(neighbour[facei]);

            faceFluxIn[facei] =
                faceAreas[facei]
                & (weightIn[facei] * (volFieldIn[own] - volFieldIn[nei]) + volFieldIn[nei]);
        }
    );

    NeoN::parallelFor(
        exec,
        {nInternalFaces, faceFluxIn.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            auto faceBCI = facei - nInternalFaces;

            faceFluxIn[facei] = bSf[faceBCI] & volFieldBc[faceBCI];
            bvalue[faceBCI] = bSf[faceBCI] & volFieldBc[faceBCI];
        }
    );

    return faceFlux;
}

}
