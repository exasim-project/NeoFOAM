// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/algorithms/pressureVelocityCoupling.hpp"
#include "Kokkos_Core.hpp"

namespace la = NeoN::la;

namespace NeoFOAM
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
                NEON_LAMBDA(const size_t bfacei) { hByABcValue[bfacei] = uBcValue[bfacei]; }
            );
        }
    }
}

nnfvcc::VolumeField<scalar> computeRAU(const PDE<Vec3>& expr)
{
    const auto& ls = expr.linearSystem();
    NF_ASSERT(
        ls.matrix().values().size() > 0,
        "computeRAU: linear system not assembled - call PDE::assemble() before reading rAU"
    );
    const auto& mesh = expr.getField().mesh();

    auto rABCs = nnfvcc::createExtrapolatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh);
    auto rAU = nnfvcc::VolumeField<scalar>(expr.exec(), "rAU", mesh, rABCs);

    NeoN::la::scaledInverseDiag(
        ls.matrix(),
        *ls.faceToMatrixAddress().get(),
        mesh.cellVolumes(),
        rAU.internalVector()
    );
    rAU.correctBoundaryConditions();
    return rAU;
}

std::tuple<nnfvcc::VolumeField<scalar>, nnfvcc::VolumeField<Vec3>>
computeRAUandHByA(const PDE<Vec3>& expr)
{
    const auto& u = expr.getField();
    const auto& mesh = u.mesh();
    const auto& ls = expr.linearSystem();
    NF_ASSERT(
        ls.matrix().values().size() > 0,
        "computeRAUandHByA: linear system not assembled - call PDE::assemble() before "
        "reading rAU/HbyA"
    );

    auto rABCs = nnfvcc::createExtrapolatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh);
    auto rAU = nnfvcc::VolumeField<scalar>(expr.exec(), "rAU", mesh, rABCs);

    auto hByABCs = nnfvcc::createExtrapolatedBCs<nnfvcc::VolumeBoundary<Vec3>>(mesh);
    auto hByA = nnfvcc::VolumeField<Vec3>(expr.exec(), "HbyA", mesh, hByABCs);

    NeoN::la::scaledInvDiagNegLUx(
        ls.matrix(),
        u.internalVector(),
        ls.rhs(),
        mesh.cellVolumes(),
        rAU.internalVector(),
        hByA.internalVector()
    );

    // Subtract processor ghost-cell contributions missing from the CSR pass above.
    // The CSR matrix only contains internal-face off-diagonals; proc ghost coupling lives in
    // offDiagonalMatrix (indexed by proc face, 0..nProcFaces-1).  Owner cell index comes from
    // the mesh topology (boundaryMesh().faceOwners() at the proc-face tail), NOT from the
    // offDiagonalMatrix sparsity which is never populated with cell indices.
    const auto nProcFaces = mesh.nProcBoundaryFaces();
    if (nProcFaces > 0)
    {
        const auto nBoundaryFaces = mesh.nBoundaryFaces();
        const auto nlValues = ls.offDiagonalMatrix().values().view();
        const auto bfOwners = mesh.boundaryMesh().faceOwners().view();
        const auto uGhostV = u.boundaryData().value().view();
        const auto rAUV = rAU.internalVector().view();
        const auto volV = mesh.cellVolumes().view();
        auto hByAV = hByA.internalVector().view();
        const auto rowOrderV = mesh.boundaryMesh().getRowOrderWriteIndex().view();

        NeoN::parallelFor(
            expr.exec(),
            {0, nProcFaces},
            NEON_LAMBDA(const NeoN::localIdx procFacei) {
                auto own = static_cast<std::size_t>(bfOwners[nBoundaryFaces + procFacei]);
                // scalar off-diagonal coupling (segregated vector-solve form): the single
                // coefficient applies to every velocity component. The future coupled Vec3
                // matrix path would index coeff per component (coeff[0..2]).
                // nlValues are stored in row-sorted order; use rowOrderV to map original
                // proc-face index to its sorted position.
                auto coeff = nlValues[rowOrderV[procFacei]];
                auto uG = uGhostV[nBoundaryFaces + procFacei];
                auto scale = rAUV[own] / volV[own];
                Kokkos::atomic_sub(&hByAV[own][0], coeff * uG[0] * scale);
                Kokkos::atomic_sub(&hByAV[own][1], coeff * uG[1] * scale);
                Kokkos::atomic_sub(&hByAV[own][2], coeff * uG[2] * scale);
            },
            "computeHbyAProcBoundary"
        );
    }

    rAU.correctBoundaryConditions();
    hByA.correctBoundaryConditions();
    return {rAU, hByA};
}


nnfvcc::VolumeField<scalar>
computeRAtU(const PDESolver<Vec3>& expr, const nnfvcc::VolumeField<scalar>& rAU)
{
    const auto& u = expr.getField();
    const auto& mesh = u.mesh();
    const auto& ls = expr.linearSystem();
    NF_ASSERT(
        ls.matrix().values().size() > 0,
        "computeRAtU: linear system not assembled - call PDESolver::assemble() before reading rAtU"
    );

    auto rAtUBCs = nnfvcc::createExtrapolatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh);
    auto rAtU = nnfvcc::VolumeField<scalar>(expr.exec(), "rAtU", mesh, rAtUBCs);

    const auto nRows = ls.matrix().nRows();

    // sumOff[cell] accumulates off-diagonal row coefficients; used below to form
    // rAtU = 1/(1/rAU + sumOff/V).
    auto sumOff = NeoN::Vector<scalar>(expr.exec(), nRows, scalar(0));

    const auto [rowOffsV, colIdxV, matrixV] = views(
        ls.matrix().sparsity()->rowOffs(),
        ls.matrix().sparsity()->colIdxs(),
        ls.matrix().values()
    );
    auto sumOffV = sumOff.view();

    NeoN::parallelFor(
        expr.exec(),
        {0, nRows},
        NEON_LAMBDA(const NeoN::localIdx rowi) {
            scalar s = scalar(0);
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                if (colIdxV[i] != rowi) s += matrixV[i];
            }
            sumOffV[rowi] = s;
        },
        "computeRAtUSumOffDiag"
    );

    // Processor ghost-cell coupling lives in offDiagonalMatrix (indexed per proc face),
    // not in the CSR sparsity. Fold it into the owner row's off-diagonal sum exactly as
    // computeRAUandHByA treats the same coupling for HbyA.
    const auto nProcFaces = mesh.nProcBoundaryFaces();
    if (nProcFaces > 0)
    {
        const auto nBoundaryFaces = mesh.nBoundaryFaces();
        const auto nlValues = ls.offDiagonalMatrix().values().view();
        const auto bfOwners = mesh.boundaryMesh().faceOwners().view();
        const auto rowOrderV = mesh.boundaryMesh().getRowOrderWriteIndex().view();

        NeoN::parallelFor(
            expr.exec(),
            {0, nProcFaces},
            NEON_LAMBDA(const NeoN::localIdx procFacei) {
                auto own = static_cast<std::size_t>(bfOwners[nBoundaryFaces + procFacei]);
                auto coeff = nlValues[rowOrderV[procFacei]];
                Kokkos::atomic_add(&sumOffV[own], coeff);
            },
            "computeRAtUProcBoundary"
        );
    }

    const auto volV = mesh.cellVolumes().view();
    const auto rAUV = rAU.internalVector().view();
    auto rAtUV = rAtU.internalVector().view();

    NeoN::parallelFor(
        expr.exec(),
        {0, nRows},
        NEON_LAMBDA(const NeoN::localIdx rowi) {
            const scalar denom = scalar(1) / rAUV[rowi] + sumOffV[rowi] / volV[rowi];
            rAtUV[rowi] = scalar(1) / denom;
        },
        "computeRAtUFinalize"
    );

    rAtU.correctBoundaryConditions();
    return rAtU;
}


void addConsistentFluxCorrection(
    nnfvcc::SurfaceField<scalar>& phiHbyA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& rAtU,
    const nnfvcc::VolumeField<scalar>& p
)
{
    const auto exec = phiHbyA.exec();
    const auto& mesh = phiHbyA.mesh();

    auto drAUBCs = nnfvcc::createExtrapolatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh);
    auto drAU = nnfvcc::VolumeField<scalar>(exec, "drAU", mesh, drAUBCs);
    {
        const auto [iRAtU, iRAU] = views(rAtU.internalVector(), rAU.internalVector());
        drAU.internalVector().apply(NEON_LAMBDA(const std::size_t celli) {
            return iRAtU[celli] - iRAU[celli];
        });
    }
    drAU.correctBoundaryConditions();

    auto linear =
        nnfvcc::SurfaceInterpolation<scalar>(exec, mesh, NeoN::TokenList({std::string("linear")}));
    auto drAUf = linear.interpolate(drAU);

    auto sng =
        nnfvcc::FaceNormalGradient<scalar>(exec, mesh, NeoN::TokenList({std::string("corrected")}));
    auto snGradP = sng.faceNormalGrad(p);

    const auto nInternalFaces = mesh.nInternalFaces();
    {
        const auto [df, sg, magSf] =
            views(drAUf.internalVector(), snGradP.internalVector(), mesh.faceAreas());
        auto phiV = phiHbyA.internalVector().view();
        NeoN::parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const NeoN::localIdx facei) {
                phiV[facei] += df[facei] * sg[facei] * magSf[facei];
            },
            "addConsistentFluxInternal"
        );
    }

    // Regular boundary faces (processor faces excluded: their interpolated drAU is not
    // populated here, and updateFaceVelocity reconstructs the proc-face flux separately).
    {
        const auto [dfB, sgB, magSfB] = views(
            drAUf.boundaryData().value(),
            snGradP.boundaryData().value(),
            mesh.boundaryMesh().faceAreas()
        );
        auto phiB = phiHbyA.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, mesh.nBoundaryFaces()},
            NEON_LAMBDA(const NeoN::localIdx bfacei) {
                phiB[bfacei] += dfB[bfacei] * sgB[bfacei] * magSfB[bfacei];
            },
            "addConsistentFluxBoundary"
        );
    }
}


void subtractConsistentHbyA(
    nnfvcc::VolumeField<Vec3>& hByA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& rAtU,
    const nnfvcc::VolumeField<scalar>& p
)
{
    auto gradP = nnfvcc::GaussGreenGrad(p.exec(), p.mesh()).grad(p);
    const auto [iHbyA, iRAU, iRAtU, iGradP] = views(
        hByA.internalVector(),
        rAU.internalVector(),
        rAtU.internalVector(),
        gradP.internalVector()
    );
    hByA.internalVector().apply(NEON_LAMBDA(const std::size_t celli) {
        return iHbyA[celli] - (iRAU[celli] - iRAtU[celli]) * iGradP[celli];
    });
    hByA.correctBoundaryConditions();
}


void updateFaceVelocity(
    const nnfvcc::SurfaceField<scalar>& predictedPhi,
    const PDE<scalar>& expr,
    nnfvcc::SurfaceField<scalar>& phi
)
{
    const auto& mesh = phi.mesh();
    const auto& p = expr.getField();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto exec = phi.exec();
    const auto [owner, neighbour, internalP] =
        views(mesh.faceOwners(), mesh.faceNeighbors(), p.internalVector());

    const auto& ls = expr.linearSystem();
    const auto rowPtrs = ls.matrix().sparsity()->rowOffs().view();
    const auto neiOffs = ls.faceToMatrixAddress()->neighbourOffset().view();
    const auto ownOffs = ls.faceToMatrixAddress()->ownerOffset().view();
    auto values = ls.matrix().values().view();
    auto [iPhi, iPredPhi] = views(phi.internalVector(), predictedPhi.internalVector());

    // Deferred non-orthogonal correction flux (OpenFOAM fvMatrix::faceFluxCorrectionPtr_).
    // The Laplacian assembly stashed the exact per-face correction it deferred to the RHS — using
    // the field as it stood at assembly time — so adding it back here gives
    // pEqn.flux() = orthogonal matrix-coefficient flux + faceFluxCorrection and div(phi) closes on
    // non-orthogonal meshes. Reusing the stored value (rather than recomputing from the post-solve
    // p) is what makes the closure exact, and avoids an extra snGrad evaluation. Null pointer ⇒
    // orthogonal / uncorrected scheme ⇒ no correction.
    const auto& ffcPtr = ls.faceFluxCorrection();
    const bool hasCorrection = (ffcPtr != nullptr) && (ffcPtr->size() == nInternalFaces);
    NeoN::Vector<scalar> noCorrection(exec, 0);
    const auto ffc = hasCorrection ? ffcPtr->view() : noCorrection.view();

    // TODO add to NEON
    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const size_t facei) {
            auto own = static_cast<std::size_t>(owner[facei]);
            auto nei = static_cast<std::size_t>(neighbour[facei]);

            auto rowNeiStart = rowPtrs[nei];
            auto rowOwnStart = rowPtrs[own];

            auto upper = values[rowNeiStart + neiOffs[facei]];
            auto lower = values[rowOwnStart + ownOffs[facei]];

            scalar corrFlux = hasCorrection ? ffc[facei] : scalar(0);
            iPhi[facei] =
                iPredPhi[facei] - (upper * internalP[nei] - lower * internalP[own]) - corrFlux;
        }
    );

    auto [bvalue, bPredValue, faceCells] = views(
        phi.boundaryData().value(),
        predictedPhi.boundaryData().value(),
        mesh.boundaryMesh().faceOwners()
    );

    const auto [mValue, rhsValue] = views(ls.boundaryMatrix(), ls.boundaryRhs());

    NeoN::parallelFor(
        exec,
        {0, static_cast<size_t>(mesh.nBoundaryFaces())},
        NEON_LAMBDA(const size_t bfacei) {
            scalar bflux =
                (rhsValue[bfacei] - mValue.values[bfacei] * internalP[faceCells[bfacei]]);
            bvalue[bfacei] = bPredValue[bfacei] - bflux;
        }
    );

    // Processor-boundary faces: proc patch sits at the tail of boundaryData().value()
    // starting at index nBoundaryFaces. Indexed directly as [0, nProcFaces) to stay
    // within the boundary-data buffer (phi.internalVector() only covers internal faces).
    const auto nProcFaces = mesh.nProcBoundaryFaces();
    if (nProcFaces > 0)
    {
        const auto nlValues = ls.offDiagonalMatrix().values().view();
        const auto pBoundV = p.boundaryData().value().view();
        const auto rowOrderV = mesh.boundaryMesh().getRowOrderWriteIndex().view();

        NeoN::parallelFor(
            exec,
            {0, nProcFaces},
            NEON_LAMBDA(const size_t procFacei) {
                auto bfacei = nBoundaryFaces + procFacei;
                auto own = static_cast<std::size_t>(faceCells[bfacei]);
                auto coupling = nlValues[rowOrderV[procFacei]];
                auto pGhost = pBoundV[bfacei];
                scalar pflux = coupling * (pGhost - internalP[own]);
                bvalue[bfacei] = bPredValue[bfacei] - pflux;
            }
        );
    }
}

void updateVelocity(
    const nnfvcc::VolumeField<Vec3>& hByA,
    const nnfvcc::VolumeField<scalar>& rAU,
    const nnfvcc::VolumeField<scalar>& p,
    nnfvcc::VolumeField<Vec3>& u,
    const nnfvcc::GradOperatorFactory<NeoN::Vec3>& gradPScheme
)
{
    // grad(p) for the velocity reconstruction honours the configured gradSchemes
    // entry for grad(p) (e.g. cellLimited); the operator is built once by the solver.
    auto gradP = gradPScheme.grad(p, NeoN::dsl::Coeff {});
    auto [iHbyA, iRAU, iGradP] =
        views(hByA.internalVector(), rAU.internalVector(), gradP.internalVector());

    u.internalVector().apply(NEON_LAMBDA(const std::size_t celli) {
        return iHbyA[celli] - iRAU[celli] * iGradP[celli];
    });
}

nnfvcc::SurfaceField<scalar> flux(const nnfvcc::VolumeField<Vec3>& volField)
{
    const auto exec = volField.exec();

    const auto& mesh = volField.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.boundaryMesh().nBoundaryFaces();
    NeoN::Input input = NeoN::TokenList({std::string("linear")});
    auto linear = nnfvcc::SurfaceInterpolation<Vec3>(exec, mesh, input);
    const auto weight = linear.weight(volField);

    auto surfaceBCs = nnfvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh);
    auto faceFlux = nnfvcc::SurfaceField<scalar>(exec, "out", mesh, surfaceBCs);

    NeoN::fill(faceFlux.internalVector(), NeoN::zero<scalar>());
    NeoN::fill(faceFlux.boundaryData().value(), NeoN::zero<scalar>());
    const auto [owner, neighbour, weightIn, faceNormals, volFieldIn, volFieldBc, bSf] = views(
        mesh.faceOwners(),
        mesh.faceNeighbors(),
        weight.internalVector(),
        mesh.faceNormals(),
        volField.internalVector(),
        volField.boundaryData().value(),
        mesh.boundaryMesh().faceNormals()
    );

    auto [faceFluxIn, bvalue] = views(faceFlux.internalVector(), faceFlux.boundaryData().value());

    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const size_t facei) {
            auto own = static_cast<std::size_t>(owner[facei]);
            auto nei = static_cast<std::size_t>(neighbour[facei]);

            faceFluxIn[facei] =
                faceNormals[facei]
                & (weightIn[facei] * (volFieldIn[own] - volFieldIn[nei]) + volFieldIn[nei]);
        }
    );

    NeoN::parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const size_t faceBCI) { bvalue[faceBCI] = bSf[faceBCI] & volFieldBc[faceBCI]; }
    );

    // Processor-boundary faces.
    // faceBCI = nBoundaryFaces + proci -> proc tail of boundaryData().value()
    const auto nProcFaces = mesh.nProcBoundaryFaces();
    const auto procFaceCells = mesh.boundaryMesh().faceOwners().view();
    const auto bndWeights = mesh.boundaryMesh().weights().view();

    NeoN::parallelFor(
        exec,
        {0, nProcFaces},
        NEON_LAMBDA(const size_t proci) {
            auto faceBCI = static_cast<size_t>(nBoundaryFaces) + proci;
            auto own = static_cast<std::size_t>(procFaceCells[faceBCI]);
            auto w = bndWeights[faceBCI];
            auto faceVal = w * (volFieldIn[own] - volFieldBc[faceBCI]) + volFieldBc[faceBCI];
            bvalue[faceBCI] = bSf[faceBCI] & faceVal;
        }
    );

    return faceFlux;
}

}
