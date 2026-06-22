// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/turbulenceModels/kOmegaSST.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "NeoFOAM/auxiliary/writers.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"

#include "wallDist.H"
#include "IOobject.H"
#include "volFields.H"

namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;

using NeoN::localIdx;
using NeoN::Tensor;
using NeoN::SymmTensor;

namespace NeoFOAM
{

namespace
{

// ---------------------------------------------------------------------------
// Fused kernel: F1, production/sp for k, production/sp for omega, and the
// internal nut update in a single pass over cells.
// ---------------------------------------------------------------------------
void kernelComputeF1AndSources(
    const NeoN::Executor& exec,
    const NeoN::Vector<scalar>& kVec,
    const NeoN::Vector<scalar>& omegaVec,
    const NeoN::Vector<scalar>& nutVec,
    const NeoN::Vector<scalar>& nuVec,
    const NeoN::Vector<scalar>& wallDistVec,
    const NeoN::Vector<Vec3>& gradKVec,
    const NeoN::Vector<Vec3>& gradOmegaVec,
    const NeoN::Vector<Tensor>& gradUVec,
    NeoN::Vector<scalar>& F1Vec,
    NeoN::Vector<scalar>& PkVec,
    NeoN::Vector<scalar>& spKVec,
    NeoN::Vector<scalar>& omegaSourceVec,
    NeoN::Vector<scalar>& spOmegaVec,
    scalar alphaK1,
    scalar alphaK2,
    scalar alphaOmega1,
    scalar alphaOmega2,
    scalar gamma1,
    scalar gamma2,
    scalar beta1,
    scalar beta2,
    scalar betaStar,
    scalar a1,
    scalar b1,
    scalar c1
)
{
    const auto
        [kV,
         omegaV,
         nutV,
         nuV,
         wallDistV,
         gradKV,
         gradOmegaV,
         gradUV,
         F1V,
         PkV,
         spKV,
         omegaSourceV,
         spOmegaV] =
            NeoN::views(
                kVec,
                omegaVec,
                nutVec,
                nuVec,
                wallDistVec,
                gradKVec,
                gradOmegaVec,
                gradUVec,
                F1Vec,
                PkVec,
                spKVec,
                omegaSourceVec,
                spOmegaVec
            );

    const scalar rootVSmall = scalar(1e-30);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(kVec.size())},
        NEON_LAMBDA(const localIdx i) {
            const scalar k_i = kV[i];
            const scalar omega_i = omegaV[i];
            const scalar nu_i = nuV[i];
            const scalar y_i = Kokkos::max(wallDistV[i], rootVSmall);
            const scalar y2_i = y_i * y_i;

            // ----- CDkOmega (cross-diffusion, clamped for F1 stability) -----
            const scalar dotGradKOmega = gradKV[i][0] * gradOmegaV[i][0]
                                       + gradKV[i][1] * gradOmegaV[i][1]
                                       + gradKV[i][2] * gradOmegaV[i][2];

            const scalar CDkOmegaPlus =
                Kokkos::max(scalar(2) * alphaOmega2 * dotGradKOmega / omega_i, scalar(1e-10));

            // ----- F1 blending (inner ↔ outer) -----
            const scalar sqrtK = Kokkos::sqrt(Kokkos::max(k_i, scalar(0)));

            const scalar arg1 = Kokkos::min(
                Kokkos::min(
                    Kokkos::max(
                        sqrtK / (betaStar * omega_i * y_i),
                        scalar(500) * nu_i / (y2_i * omega_i)
                    ),
                    scalar(4) * alphaOmega2 * k_i / (CDkOmegaPlus * y2_i)
                ),
                scalar(10)
            );
            const scalar arg14 = arg1 * arg1 * arg1 * arg1;
            F1V[i] = Kokkos::tanh(arg14);
            const scalar F1_i = F1V[i];

            // ----- F2 (for nut correction) -----
            const scalar arg2 = Kokkos::min(
                Kokkos::max(
                    scalar(2) * sqrtK / (betaStar * omega_i * y_i),
                    scalar(500) * nu_i / (y2_i * omega_i)
                ),
                scalar(100)
            );
            const scalar F2_i = Kokkos::tanh(arg2 * arg2);

            // ----- S2 = 2*|symm(gradU)|^2 and GbyNu0 = gradU && devTwoSymm(gradU) -----
            const Tensor& g = gradUV[i];

            scalar normSq = scalar(0);
            scalar dotTrans = scalar(0);
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    normSq += g(r, c) * g(r, c);
                    dotTrans += g(r, c) * g(c, r);
                }
            }
            const scalar divU = g(0, 0) + g(1, 1) + g(2, 2);

            const scalar S2_i = normSq + dotTrans;
            const scalar GbyNu0_i = S2_i - (scalar(2) / scalar(3)) * divU * divU;

            // ----- nut (for production G = nut * GbyNu0) -----
            const scalar sqrtS2 = Kokkos::sqrt(Kokkos::max(S2_i, scalar(0)));
            const scalar nut_i = nutV[i]; // use OLD nut for G (consistent with OF sequence)

            // ----- Blended coefficients -----
            const scalar gamma_i = F1_i * (gamma1 - gamma2) + gamma2;
            const scalar beta_i = F1_i * (beta1 - beta2) + beta2;

            // ----- k equation sources -----
            // G = nut * GbyNu0, Pk = min(G, c1*betaStar*k*omega)
            const scalar G_i = nut_i * GbyNu0_i;
            PkV[i] = Kokkos::min(G_i, c1 * betaStar * k_i * omega_i);

            // betaStar*omega as implicit destruction for k
            spKV[i] = betaStar * omega_i;

            // ----- omega equation sources -----
            // Bounded GbyNu for omega production
            const scalar GbyNuBound_i = Kokkos::min(
                GbyNu0_i,
                (c1 / a1) * betaStar * omega_i * Kokkos::max(a1 * omega_i, b1 * F2_i * sqrtS2)
            );
            omegaSourceV[i] = gamma_i * GbyNuBound_i;

            // beta*omega as implicit destruction for omega (base)
            spOmegaV[i] = beta_i * omega_i;

            // Cross-diffusion: (1-F1)*CDkOmega (actual, may be negative)
            const scalar CDkOmegaActual = scalar(2) * alphaOmega2 * dotGradKOmega / omega_i;
            const scalar crossSource = (scalar(1) - F1_i) * CDkOmegaActual;

            // Positive cross-source → explicit; negative → implicit sink for stability
            omegaSourceV[i] += Kokkos::max(crossSource, scalar(0));
            spOmegaV[i] += Kokkos::max(-crossSource / omega_i, scalar(0));
        },
        "kOmegaSST::computeF1AndSources"
    );
}

// ---------------------------------------------------------------------------
// Kernel: update internal nut from new k/omega using current gradU_ for S2.
// ---------------------------------------------------------------------------
void kernelCorrectNutInternal(
    const NeoN::Executor& exec,
    const NeoN::Vector<scalar>& kVec,
    const NeoN::Vector<scalar>& omegaVec,
    const NeoN::Vector<scalar>& wallDistVec,
    const NeoN::Vector<scalar>& nuVec,
    const NeoN::Vector<Tensor>& gradUVec,
    NeoN::Vector<scalar>& nutVec,
    scalar betaStar,
    scalar a1,
    scalar b1
)
{
    const auto [kV, omegaV, wallDistV, nuV, gradUV, nutV_] =
        NeoN::views(kVec, omegaVec, wallDistVec, nuVec, gradUVec, nutVec);

    const scalar rootVSmall = scalar(1e-30);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(kVec.size())},
        NEON_LAMBDA(const localIdx i) {
            const scalar k_i = kV[i];
            const scalar omega_i = omegaV[i];
            const scalar nu_i = nuV[i];
            const scalar y_i = Kokkos::max(wallDistV[i], rootVSmall);
            const scalar y2_i = y_i * y_i;

            // F2
            const scalar sqrtK = Kokkos::sqrt(Kokkos::max(k_i, scalar(0)));
            const scalar arg2 = Kokkos::min(
                Kokkos::max(
                    scalar(2) * sqrtK / (betaStar * omega_i * y_i),
                    scalar(500) * nu_i / (y2_i * omega_i)
                ),
                scalar(100)
            );
            const scalar F2_i = Kokkos::tanh(arg2 * arg2);

            // S2 from current gradU
            const Tensor& g = gradUV[i];
            scalar normSq = scalar(0);
            scalar dotTrans = scalar(0);
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    normSq += g(r, c) * g(r, c);
                    dotTrans += g(r, c) * g(c, r);
                }
            }
            const scalar S2_i = normSq + dotTrans;
            const scalar sqrtS2 = Kokkos::sqrt(Kokkos::max(S2_i, scalar(0)));

            // nut = a1*k / max(a1*omega, b1*F2*sqrt(S2))
            nutV_[i] = a1 * k_i / Kokkos::max(a1 * omega_i, b1 * F2_i * sqrtS2);
        },
        "kOmegaSST::correctNutInternal"
    );
}

// ---------------------------------------------------------------------------
// Kernel: compute blended diffusivities on faces.
// DkEffF    = (F1*(alphaK1-alphaK2) + alphaK2)*nutF + nuF
// DomegaEffF = (F1*(alphaO1-alphaO2) + alphaO2)*nutF + nuF
// nuEffF     = nutF + nuF
// ---------------------------------------------------------------------------
void kernelCalcDiffusivities(
    const NeoN::Executor& exec,
    const NeoN::Vector<scalar>& surfNuVec,
    const NeoN::Vector<scalar>& surfNutVec,
    const NeoN::Vector<scalar>& surfF1Vec,
    NeoN::Vector<scalar>& nuEffVec,
    NeoN::Vector<scalar>& DkEffVec,
    NeoN::Vector<scalar>& DomegaEffVec,
    scalar alphaK1,
    scalar alphaK2,
    scalar alphaOmega1,
    scalar alphaOmega2,
    std::string label
)
{
    const auto [nuF, nutF, F1F, nuEffF, DkF, DomF] =
        NeoN::views(surfNuVec, surfNutVec, surfF1Vec, nuEffVec, DkEffVec, DomegaEffVec);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(nuEffVec.size())},
        NEON_LAMBDA(const localIdx f) {
            const scalar alphaK = F1F[f] * (alphaK1 - alphaK2) + alphaK2;
            const scalar alphaOmega = F1F[f] * (alphaOmega1 - alphaOmega2) + alphaOmega2;

            nuEffF[f] = nuF[f] + nutF[f];
            DkF[f] = alphaK * nutF[f] + nuF[f];
            DomF[f] = alphaOmega * nutF[f] + nuF[f];
        },
        std::move(label)
    );
}

void kernelDevRhoReff(
    const NeoN::Executor& exec,
    const NeoN::Vector<Tensor>& gradUB,
    const NeoN::Vector<scalar>& nuEffB,
    NeoN::Vector<SymmTensor>& result
)
{
    const auto [gV, nuEffV, resV] = NeoN::views(gradUB, nuEffB, result);
    const localIdx nBF = static_cast<localIdx>(gradUB.size());

    NeoN::parallelFor(
        exec,
        {0, nBF},
        NEON_LAMBDA(const localIdx bf) {
            resV[bf] = NeoN::symm(NeoN::twoSymm(gV[bf])).dev2() * (-nuEffV[bf]);
        },
        "kOmegaSST::devRhoReff::boundary"
    );
}

} // namespace

// ============================================================
// Constructor
// ============================================================

KOmegaSST::KOmegaSST(
    const NeoN::Executor& exec,
    const NeoN::UnstructuredMesh& mesh,
    const nnfvcc::VolumeField<scalar>& nu,
    const nnfvcc::VolumeField<scalar>& wallDist
)
    : exec_(exec)
    , mesh_(mesh)
    , nu_(nu)
    , wallDist_(wallDist)
    , nearWallDist_(
          exec,
          "nearWallDist",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , surfNu_(
          exec,
          "surfNu",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , gradU_(
          exec,
          "gradU",
          mesh,
          fvcc::createCalculatedProcBCs<nnfvcc::VolumeBoundary<Tensor>>(mesh)
      )
    , gradK_(exec, "gradK", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<Vec3>>(mesh))
    , gradOmega_(
          exec,
          "gradOmega",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<Vec3>>(mesh)
      )
    , F1_(exec, "F1", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh))
    , Pk_(exec, "Pk", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh))
    , spK_(exec, "spK", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh))
    , omegaSource_(
          exec,
          "omegaSource",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , spOmega_(
          exec,
          "spOmega",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , surfNut_(
          exec,
          "surfNut",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , surfF1_(
          exec,
          "surfF1",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , nuEff_(exec, "nuEff", mesh, fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh))
    , DkEffF_(exec, "DkEff", mesh, fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh))
    , DomegaEffF_(
          exec,
          "DomegaEff",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , gradOp_(exec, mesh)
    , surfInterp_(exec, mesh, NeoN::TokenList({std::string("linear")}))
    , coeffs_()
{
    surfInterp_.interpolate(nu_, surfNu_);

    // Populate nearWallDist_'s boundary face values by copying wallDist_'s
    // adjacent-cell internal value. This is the same per-face "y" that
    // OpenFOAM's turbulenceModel::y()[patchi] exposes (and what the
    // wall-function unit tests construct explicitly). Mesh is static, so
    // doing it once at construction is sufficient.
    {
        const auto wdInternal = wallDist_.internalVector().view();
        const auto faceOwners = mesh_.boundaryMesh().faceOwners().view();
        auto nwdBoundary = nearWallDist_.boundaryData().value().view();
        const auto nBoundaryFaces = static_cast<NeoN::localIdx>(nwdBoundary.size());
        NeoN::parallelFor(
            exec_,
            {0, nBoundaryFaces},
            NEON_LAMBDA(const NeoN::localIdx i) { nwdBoundary[i] = wdInternal[faceOwners[i]]; },
            "kOmegaSST::initNearWallDistBoundary"
        );
    }
}

// ============================================================
// Public interface
// ============================================================

void KOmegaSST::validate(
    const nnfvcc::VolumeField<Vec3>& U,
    const nnfvcc::VolumeField<scalar>& k,
    const nnfvcc::VolumeField<scalar>& omega,
    nnfvcc::VolumeField<scalar>& nut
)
{
    gradOp_.gradTensor(U, gradU_);
    gradOp_.grad(k, gradK_);
    gradOp_.grad(omega, gradOmega_);

    computeF1AndSources(k, omega, nut, gradK_, gradOmega_, gradU_);

    correctNutInternal(k, omega, nut);
    // Wall-function BCs on nut (Spalding) need (U, nu, nearWallDist). The
    // cell-centered wallDist field with zeroGradient-style boundary values
    // gives the cell-to-wall distance at wall faces, which is what the
    // wall-function kernels consume as `y`. Same context is reused for
    // omega and k below (kqRWallFunction ignores it).
    fvcc::BoundaryContext ctx;
    ctx.insert("U", U);
    ctx.insert("k", k);
    ctx.insert("nu", nu_);
    ctx.insert("nearWallDist", nearWallDist_);
    nut.correctBoundaryConditions(ctx);

    calcDiffusivities(nut);
}

void KOmegaSST::correct(
    const nnfvcc::VolumeField<Vec3>& U,
    nnfvcc::SurfaceField<scalar>& phi,
    nnfvcc::VolumeField<scalar>& k,
    nnfvcc::VolumeField<scalar>& omega,
    nnfvcc::VolumeField<scalar>& nut,
    RunTime& rt
)
{
    // Compute gradients once — reused for production, CDkOmega, and correctNut
    gradOp_.gradTensor(U, gradU_);
    gradU_.correctBoundaryConditions();
    gradOp_.grad(k, gradK_);
    gradOp_.grad(omega, gradOmega_);

    // Build the BoundaryContext once and reuse for the three correctBC calls
    // below. omegaWallFunction reads k+nu+nearWallDist;
    // nutUSpaldingWallFunction reads U+nu+nearWallDist; kqRWallFunction
    // ignores the context (pure zero-gradient). Passing extras is harmless.
    fvcc::BoundaryContext ctx;
    ctx.insert("U", U);
    ctx.insert("k", k);
    ctx.insert("nu", nu_);
    ctx.insert("nearWallDist", nearWallDist_);

    // Compute F1, sources, and sp coefficients using OLD nut
    computeF1AndSources(k, omega, nut, gradK_, gradOmega_, gradU_);

    // Refresh face diffusivities (need DkEffF_ and DomegaEffF_ for PDEs below)
    calcDiffusivities(nut);

    // OF feedback: overwrite Pk_ at wall-function cells with the omega wall
    // function's G formula. Mirrors omegaWallFunctionFvPatchScalarField::
    // calculate() (omega.C:319-333) + updateCoeffs() (omega.C:518-524):
    //   G[wall_cell] = (ν_t_w + ν_w) · |∂U/∂n|_w · C_µ^0.25 · √k / (κ · y)
    // The standard formula Pk = nut · GbyNu0 underestimates near-wall
    // production at refined meshes because the cell-center gradU is a poor
    // proxy for the true wall stress; the wall-function form uses the log-law
    // gradient directly. Without this overwrite the k equation lacks the
    // source it needs to balance the wall-function ω boundary value.
    //
    // Identifies wall-function patches by name match on omega's BC list;
    // BINOMIAL blender always applies the formula on every face (omega.C:262
    // STEPWISE gates on yPlus, not modelled here).
    {
        const scalar Cmu25 = Kokkos::pow(coeffs_.betaStar, scalar(0.25));
        const scalar kappa_ = scalar(0.41); // matches omegaWallFunction default
        const auto& omegaBCs = omega.boundaryConditions();
        const auto faceOwnersV = mesh_.boundaryMesh().faceOwners().view();
        const auto deltaCoeffsV = mesh_.boundaryMesh().deltaCoeffs().view();
        const auto uInternalV = U.internalVector().view();
        const auto uBoundaryV = U.boundaryData().value().view();
        const auto nuBoundaryV = nu_.boundaryData().value().view();
        const auto nutBoundaryV = nut.boundaryData().value().view();
        const auto yBoundaryV = nearWallDist_.boundaryData().value().view();
        const auto kInternalV = k.internalVector().view();
        auto pkInternalV = Pk_.internalVector().view();

        for (NeoN::localIdx patchID = 0; patchID < static_cast<NeoN::localIdx>(omegaBCs.size());
             ++patchID)
        {
            if (omegaBCs[static_cast<size_t>(patchID)].name() != "omegaWallFunction")
            {
                continue;
            }
            const auto [start, end] = omega.boundaryData().range(patchID);
            NeoN::parallelFor(
                exec_,
                {start, end},
                NEON_LAMBDA(const NeoN::localIdx i) {
                    const auto owner = faceOwnersV[i];
                    const NeoN::Vec3 uOwn = uInternalV[owner];
                    const NeoN::Vec3 uWall = uBoundaryV[i];
                    const scalar deltaInv = deltaCoeffsV[i];
                    // snGrad(U) = (U_face - U_owner) / delta — same as upstream
                    // fvPatchVectorField::snGrad() for fixedValue / noSlip BCs.
                    const NeoN::Vec3 snGradU = (uWall - uOwn) * deltaInv;
                    const scalar magGradUw = NeoN::mag(snGradU);
                    const scalar nuw = nuBoundaryV[i];
                    const scalar nutw = nutBoundaryV[i];
                    const scalar y = yBoundaryV[i];
                    const scalar kc = Kokkos::max(kInternalV[owner], scalar(0));
                    const scalar gWall =
                        (nutw + nuw) * magGradUw * Cmu25 * Kokkos::sqrt(kc) / (kappa_ * y);
                    // Overwrite — matches OF's G[celli] = G0[celli] assignment.
                    pkInternalV[owner] = gWall;
                },
                "kOmegaSST::omegaWFGFeedback"
            );
        }
    }

    // ----- omega equation -----
    PDESolver<scalar> omegaEqn(
        dsl::imp::ddt(omega) + dsl::imp::div(phi, omega) - dsl::imp::laplacian(DomegaEffF_, omega)
            + dsl::imp::source(spOmega_, omega) - dsl::exp::source(omegaSource_),
        omega,
        rt
    );
    omegaEqn.solve();

    // Bound omega > 0
    {
        auto omegaView = omega.internalVector().view();
        NeoN::parallelFor(
            exec_,
            {0, static_cast<localIdx>(omega.internalVector().size())},
            NEON_LAMBDA(const localIdx i) {
                omegaView[i] = Kokkos::max(omegaView[i], scalar(1e-10));
            },
            "kOmegaSST::boundOmega"
        );
        omega.correctBoundaryConditions(ctx);
    }

    // Update spK_ with new omega so the k equation uses the post-omega-solve
    // destruction coefficient — matching OpenFOAM's kOmegaSSTBase::correct() sequence
    // where epsilonByk(F1, gradU) = betaStar*omega_() references the updated omega.
    {
        const scalar betaStar = coeffs_.betaStar;
        auto omegaView = omega.internalVector().view();
        auto spKView = spK_.internalVector().view();
        NeoN::parallelFor(
            exec_,
            {0, static_cast<localIdx>(omega.internalVector().size())},
            NEON_LAMBDA(const localIdx i) { spKView[i] = betaStar * omegaView[i]; },
            "kOmegaSST::updateSpKFromNewOmega"
        );
    }

    // ----- k equation -----
    PDESolver<scalar> kEqn(
        dsl::imp::ddt(k) + dsl::imp::div(phi, k) - dsl::imp::laplacian(DkEffF_, k)
            + dsl::imp::source(spK_, k) - dsl::exp::source(Pk_),
        k,
        rt
    );
    kEqn.solve();

    // Bound k >= 0
    {
        auto kView = k.internalVector().view();
        NeoN::parallelFor(
            exec_,
            {0, static_cast<localIdx>(k.internalVector().size())},
            NEON_LAMBDA(const localIdx i) { kView[i] = Kokkos::max(kView[i], scalar(0)); },
            "kOmegaSST::boundK"
        );
        k.correctBoundaryConditions(ctx);
    }

    // Update nut from new k, omega (uses gradU_ from start of this timestep for S2)
    correctNutInternal(k, omega, nut);
    nut.correctBoundaryConditions(ctx);

    // Refresh surface diffusivities with new nut
    calcDiffusivities(nut);
}

nnfvcc::SurfaceField<scalar>& KOmegaSST::nuEff() { return nuEff_; }

nnfvcc::SurfaceField<scalar>& KOmegaSST::DkEff() { return DkEffF_; }

nnfvcc::SurfaceField<scalar>& KOmegaSST::DomegaEff() { return DomegaEffF_; }

const nnfvcc::VolumeField<Tensor>& KOmegaSST::gradU() const { return gradU_; }

NeoN::Vector<SymmTensor> KOmegaSST::devRhoReff() const
{
    const localIdx nBF = static_cast<localIdx>(gradU_.boundaryData().value().size());
    NeoN::Vector<SymmTensor> result(exec_, nBF, NeoN::zero<SymmTensor>());
    kernelDevRhoReff(exec_, gradU_.boundaryData().value(), nuEff_.boundaryData().value(), result);
    return result;
}

// ============================================================
// Public physics helpers
// ============================================================

void KOmegaSST::computeF1AndSources(
    const nnfvcc::VolumeField<scalar>& k,
    const nnfvcc::VolumeField<scalar>& omega,
    const nnfvcc::VolumeField<scalar>& nut,
    const nnfvcc::VolumeField<Vec3>& gradK,
    const nnfvcc::VolumeField<Vec3>& gradOmega,
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU
)
{
    kernelComputeF1AndSources(
        exec_,
        k.internalVector(),
        omega.internalVector(),
        nut.internalVector(),
        nu_.internalVector(),
        wallDist_.internalVector(),
        gradK.internalVector(),
        gradOmega.internalVector(),
        gradU.internalVector(),
        F1_.internalVector(),
        Pk_.internalVector(),
        spK_.internalVector(),
        omegaSource_.internalVector(),
        spOmega_.internalVector(),
        coeffs_.alphaK1,
        coeffs_.alphaK2,
        coeffs_.alphaOmega1,
        coeffs_.alphaOmega2,
        coeffs_.gamma1,
        coeffs_.gamma2,
        coeffs_.beta1,
        coeffs_.beta2,
        coeffs_.betaStar,
        coeffs_.a1,
        coeffs_.b1,
        coeffs_.c1
    );
}

void KOmegaSST::correctNutInternal(
    const nnfvcc::VolumeField<scalar>& k,
    const nnfvcc::VolumeField<scalar>& omega,
    nnfvcc::VolumeField<scalar>& nut
) const
{
    kernelCorrectNutInternal(
        exec_,
        k.internalVector(),
        omega.internalVector(),
        wallDist_.internalVector(),
        nu_.internalVector(),
        gradU_.internalVector(),
        nut.internalVector(),
        coeffs_.betaStar,
        coeffs_.a1,
        coeffs_.b1
    );
}

void KOmegaSST::calcDiffusivities(const nnfvcc::VolumeField<scalar>& nut)
{
    surfInterp_.interpolate(nut, surfNut_);
    surfInterp_.interpolate(F1_, surfF1_);

    kernelCalcDiffusivities(
        exec_,
        surfNu_.internalVector(),
        surfNut_.internalVector(),
        surfF1_.internalVector(),
        nuEff_.internalVector(),
        DkEffF_.internalVector(),
        DomegaEffF_.internalVector(),
        coeffs_.alphaK1,
        coeffs_.alphaK2,
        coeffs_.alphaOmega1,
        coeffs_.alphaOmega2,
        "kOmegaSST::calcDiffusivities::internal"
    );
    kernelCalcDiffusivities(
        exec_,
        surfNu_.boundaryData().value(),
        surfNut_.boundaryData().value(),
        surfF1_.boundaryData().value(),
        nuEff_.boundaryData().value(),
        DkEffF_.boundaryData().value(),
        DomegaEffF_.boundaryData().value(),
        coeffs_.alphaK1,
        coeffs_.alphaK2,
        coeffs_.alphaOmega1,
        coeffs_.alphaOmega2,
        "kOmegaSST::calcDiffusivities::boundary"
    );
}

namespace
{

nnfvcc::VolumeField<scalar> buildWallDistKOmega(const NeoN::Executor& exec, MeshAdapter& mesh)
{
    Foam::wallDist y(mesh);
    return NeoFOAM::constructFrom(exec, mesh.nfMesh(), y.y());
}

Foam::volScalarField readOFScalarField(MeshAdapter& mesh, const std::string& fieldName)
{
    return Foam::volScalarField(
        Foam::IOobject(
            fieldName,
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        mesh
    );
}

} // namespace

KOmegaSSTModel::KOmegaSSTModel(RunTime& rt, const nnfvcc::VolumeField<scalar>& nu)
    : nu_(nu)
    , wallDist_(buildWallDistKOmega(rt.exec, rt.mesh))
    , model_(rt.exec, rt.nfMesh, nu_, wallDist_)
{
    auto& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");

    k_ = &NeoFOAM::constructAndRegister(vc, rt, readOFScalarField(rt.mesh, "k"), true);
    omega_ = &NeoFOAM::constructAndRegister(vc, rt, readOFScalarField(rt.mesh, "omega"), true);
    nut_ = &NeoFOAM::constructAndRegister(vc, rt, readOFScalarField(rt.mesh, "nut"), false);

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    for (const auto* f : {"k", "omega", "kFinal", "omegaFinal"})
    {
        if (solverDict.isDict(f))
        {
            solverDict.subDict(f) = mapFvSolution(solverDict.subDict(f));
        }
    }
}

void KOmegaSSTModel::validate(const nnfvcc::VolumeField<Vec3>& U)
{
    model_.validate(U, *k_, *omega_, *nut_);
}

void KOmegaSSTModel::correct(
    const nnfvcc::VolumeField<Vec3>& U,
    nnfvcc::SurfaceField<scalar>& phi,
    RunTime& rt
)
{
    model_.correct(U, phi, *k_, *omega_, *nut_, rt);
}

nnfvcc::SurfaceField<scalar>& KOmegaSSTModel::nuEff() { return model_.nuEff(); }

const nnfvcc::VolumeField<scalar>& KOmegaSSTModel::nut() const { return *nut_; }

const nnfvcc::VolumeField<NeoN::Tensor>& KOmegaSSTModel::gradU() const { return model_.gradU(); }

void KOmegaSSTModel::rotateOldTimes()
{
    fvcc::rotateOldTimes(*k_);
    fvcc::rotateOldTimes(*omega_);
}

void KOmegaSSTModel::write(MeshAdapter& mesh) const
{
    NeoFOAM::write(*k_, mesh);
    NeoFOAM::write(*omega_, mesh);
    NeoFOAM::write(*nut_, mesh);
}

} // namespace NeoFOAM
