// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/turbulenceModels/kOmegaSST.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "NeoFOAM/auxiliary/writers.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"
#include "NeoFOAM/fvcc/boundary/volume/omegaWallFunction.hpp"

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

namespace detail
{
scalar reportBounding(
    const NeoN::Executor& exec,
    const nnfvcc::VolumeField<scalar>& field,
    const std::string& name,
    scalar lowerBound,
    bool isDistributed
);
void boundLowerSmoothRepair(
    const NeoN::Executor& exec,
    nnfvcc::VolumeField<scalar>& field,
    const NeoN::UnstructuredMesh& mesh,
    const nnfvcc::SurfaceInterpolation<scalar>& surfInterp,
    nnfvcc::VolumeField<scalar>& floored,
    nnfvcc::SurfaceField<scalar>& surfFloored,
    NeoN::Vector<scalar>& sumFaceArea,
    bool& sumFaceAreaBuilt,
    scalar lowerBound
);
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
);
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
);
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
);
void kernelDevRhoReff(
    const NeoN::Executor& exec,
    const NeoN::Vector<Tensor>& gradUB,
    const NeoN::Vector<scalar>& nuEffB,
    NeoN::Vector<SymmTensor>& result
);
void initNearWallDistBoundary(
    const NeoN::Executor& exec,
    const nnfvcc::VolumeField<scalar>& wallDist,
    const NeoN::UnstructuredMesh& mesh,
    nnfvcc::VolumeField<scalar>& nearWallDist
);
} // namespace detail

namespace
{
inline constexpr scalar KOSST_OMEGA_MIN = scalar(1e-10);
inline constexpr scalar KOSST_NUT_MAX = scalar(1e6);
inline constexpr scalar KOSST_OMEGA_WALL_MAX =
    NeoN::finiteVolume::cellCentred::volumeBoundary::detail::OMEGA_WF_OMEGA_MAX;
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
    , F1_(exec, "F1", mesh, fvcc::createExtrapolatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh))
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
    , omegaWallValue_(exec, static_cast<NeoN::localIdx>(mesh.nCells()), scalar(0))
    , omegaWallMask_(exec, static_cast<NeoN::localIdx>(mesh.nCells()), scalar(0))
    , cornerWeight_(exec, static_cast<NeoN::localIdx>(mesh.nCells()), scalar(0))
    , boundFloored_(
          exec,
          "omegaBoundFloored",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , surfBoundFloored_(
          exec,
          "surfBoundFloored",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , sumFaceArea_(exec, static_cast<NeoN::localIdx>(mesh.nCells()), scalar(0))
{
    surfInterp_.interpolate(nu_, surfNu_);

    // Populate nearWallDist_'s boundary face values by copying wallDist_'s
    // adjacent-cell internal value. This is the same per-face "y" that
    // OpenFOAM's turbulenceModel::y()[patchi] exposes (and what the
    // wall-function unit tests construct explicitly). Mesh is static, so
    // doing it once at construction is sufficient.
    // Done via a free function because NVCC forbids NEON_LAMBDA in a constructor body.
    detail::initNearWallDistBoundary(exec_, wallDist_, mesh_, nearWallDist_);

    // Both surfInterp_.interpolate and initNearWallDistBoundary dispatch GPU kernels
    // asynchronously.  Fence here so that the constructor's post-condition holds:
    // surfNu_ and nearWallDist_ contain fully-written results before any subsequent
    // code (e.g. KOmegaSSTModel::KOmegaSSTModel constructAndRegister calls) begins.
    NeoN::fence(exec_);
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
    //
    // The same loop also builds the omega wall-cell PIN: for each wall face it writes the
    // blended viscous/log omega into omegaWallValue_[owner] and marks omegaWallMask_[owner].
    // Passed to omegaEqn.setConstraints() below, this hard-pins the near-wall cell omega during
    // the solve (OpenFOAM's omegaWallFunction::manipulateMatrix(setValues) equivalent). Without
    // it the near-wall omega is governed only by the diffusion/destruction balance, stays too
    // small, and nut = a1*k/max(a1*omega, ...) blows up at the wall -> divergence.
    bool anyOmegaWF = false;
    {
        const scalar Cmu25 = Kokkos::pow(coeffs_.betaStar, scalar(0.25));
        const scalar kappa_ = scalar(0.41); // matches omegaWallFunction default
        const scalar beta1 = coeffs_.beta1;
        const scalar omegaWallMax = KOSST_OMEGA_WALL_MAX; // device-capture copy
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
        const auto nCells = static_cast<NeoN::localIdx>(mesh_.nCells());

        // One-time: build cornerWeight_[c] = 1/(number of omegaWallFunction faces touching cell
        // c), 0 for non-wall cells. The wall topology is static so this is computed once. Mirrors
        // OpenFOAM omegaWallFunction::createAveragingWeights (omega.C:71-126).
        if (!cornerWeightsBuilt_)
        {
            NeoN::fill(cornerWeight_, scalar(0));
            auto cwBuild = cornerWeight_.view();
            for (NeoN::localIdx patchID = 0; patchID < static_cast<NeoN::localIdx>(omegaBCs.size());
                 ++patchID)
            {
                if (omegaBCs[static_cast<size_t>(patchID)].name() != "omegaWallFunction") continue;
                const auto [start, end] = omega.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec_,
                    {start, end},
                    NEON_LAMBDA(const NeoN::localIdx i) {
                        Kokkos::atomic_add(&cwBuild[faceOwnersV[i]], scalar(1));
                    },
                    "kOmegaSST::omegaWFCountFaces"
                );
            }
            NeoN::parallelFor(
                exec_,
                {0, nCells},
                NEON_LAMBDA(const NeoN::localIdx c) {
                    if (cwBuild[c] > scalar(0)) cwBuild[c] = scalar(1) / cwBuild[c];
                },
                "kOmegaSST::omegaWFInvertCount"
            );
            cornerWeightsBuilt_ = true;
        }
        const auto cornerWeightV = cornerWeight_.view();

        // Per-step reset of the accumulators: the pin mask, the pinned value (now ACCUMULATED, so
        // it must start at zero), and Pk_ at the wall cells (computeF1AndSources wrote the bulk
        // production there; the wall log-law production replaces it via the weighted sum below).
        NeoN::fill(omegaWallMask_, scalar(0));
        NeoN::fill(omegaWallValue_, scalar(0));
        auto omegaWallValueV = omegaWallValue_.view();
        auto omegaWallMaskV = omegaWallMask_.view();
        NeoN::parallelFor(
            exec_,
            {0, nCells},
            NEON_LAMBDA(const NeoN::localIdx c) {
                if (cornerWeightV[c] > scalar(0)) pkInternalV[c] = scalar(0);
            },
            "kOmegaSST::omegaWFZeroWallPk"
        );
        NeoN::fence(exec_); // zeroing must complete before the atomic accumulation below

        for (NeoN::localIdx patchID = 0; patchID < static_cast<NeoN::localIdx>(omegaBCs.size());
             ++patchID)
        {
            if (omegaBCs[static_cast<size_t>(patchID)].name() != "omegaWallFunction")
            {
                continue;
            }
            anyOmegaWF = true;
            const auto [start, end] = omega.boundaryData().range(patchID);
            NeoN::parallelFor(
                exec_,
                {start, end},
                NEON_LAMBDA(const NeoN::localIdx i) {
                    const auto owner = faceOwnersV[i];
                    const scalar cw = cornerWeightV[owner]; // 1/(num wall faces on this cell)
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

                    // Blended omega (same formula as the omegaWallFunction BC, omega.C:218-234):
                    // viscous 6*nu/(beta1*y^2), log sqrt(k)/(Cmu^0.25*kappa*y), BINOMIAL n=2 ->
                    // sqrt(vis^2 + log^2), clamped to the SAME OMEGA_WF_OMEGA_MAX the BC uses so
                    // the wall face value and this cell pin stay identical.
                    const scalar ySafe = Kokkos::max(y, scalar(1e-30));
                    const scalar wVis = scalar(6) * nuw / (beta1 * ySafe * ySafe);
                    const scalar wLog = Kokkos::sqrt(kc) / (Cmu25 * kappa_ * ySafe);
                    const scalar wOmega =
                        Kokkos::min(Kokkos::sqrt(wVis * wVis + wLog * wLog), omegaWallMax);

                    // Corner-weighted accumulation (OF omegaWallFunction::calculate +
                    // createAveragingWeights): a cell on N wall faces gets the weighted MEAN of its
                    // per-face G and omega (weight 1/N each), not the last face. atomic_add makes
                    // the multi-face (corner) case deterministic — the old plain '=' raced.
                    Kokkos::atomic_add(&pkInternalV[owner], cw * gWall);
                    Kokkos::atomic_add(&omegaWallValueV[owner], cw * wOmega);
                    omegaWallMaskV[owner] = scalar(1);
                },
                "kOmegaSST::omegaWFGFeedback"
            );
        }
    }

    // ----- omega equation -----
    auto omegaEqn = PDESolver<scalar>(
        dsl::imp::ddt(omega) + dsl::imp::div(phi, omega) - dsl::imp::laplacian(DomegaEffF_, omega)
            + dsl::imp::source(spOmega_, omega) - dsl::exp::source(omegaSource_),
        omega,
        rt
    );
    // Hard-pin the near-wall cells to the blended wall omega built above (the missing
    // omegaWallFunction setValues equivalent). Skipped when no omegaWallFunction patch exists.
    if (anyOmegaWF)
    {
        omegaEqn.setConstraints(omegaWallMask_, omegaWallValue_);
    }
    omegaEqn.solve();

    // Bound omega >= omegaMin, mirroring OpenFOAM's Foam::bound(): report, then on the steps that
    // actually dipped below the floor replace the negative cells with the smoother neighbourhood
    // average (boundLowerSmoothRepair) instead of a hard clip. The hard clip left negative omega
    // spikes that amplified into the omega blow-up / SIGFPE seen in the pMG parameter study.
    {
        const scalar gMin = detail::reportBounding(
            exec_,
            omega,
            "omega",
            KOSST_OMEGA_MIN,
            mesh_.boundaryMesh().isDistributed()
        );
        if (gMin < KOSST_OMEGA_MIN)
        {
            detail::boundLowerSmoothRepair(
                exec_,
                omega,
                mesh_,
                surfInterp_,
                boundFloored_,
                surfBoundFloored_,
                sumFaceArea_,
                sumFaceAreaBuilt_,
                KOSST_OMEGA_MIN
            );
        }
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
    auto kEqn = PDESolver<scalar>(
        dsl::imp::ddt(k) + dsl::imp::div(phi, k) - dsl::imp::laplacian(DkEffF_, k)
            + dsl::imp::source(spK_, k) - dsl::exp::source(Pk_),
        k,
        rt
    );
    kEqn.solve();

    // Bound k >= 0, mirroring OpenFOAM's Foam::bound() with the same smoother repair as omega.
    {
        const scalar gMin =
            detail::reportBounding(exec_, k, "k", scalar(0), mesh_.boundaryMesh().isDistributed());
        if (gMin < scalar(0))
        {
            detail::boundLowerSmoothRepair(
                exec_,
                k,
                mesh_,
                surfInterp_,
                boundFloored_,
                surfBoundFloored_,
                sumFaceArea_,
                sumFaceAreaBuilt_,
                scalar(0)
            );
        }
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

void KOmegaSST::updateGradU(const nnfvcc::VolumeField<Vec3>& U)
{
    gradOp_.gradTensor(U, gradU_);
    gradU_.correctBoundaryConditions();
}

NeoN::Vector<SymmTensor> KOmegaSST::devRhoReff() const
{
    const localIdx nBF = static_cast<localIdx>(gradU_.boundaryData().value().size());
    NeoN::Vector<SymmTensor> result(exec_, nBF, NeoN::zero<SymmTensor>());
    detail::kernelDevRhoReff(
        exec_,
        gradU_.boundaryData().value(),
        nuEff_.boundaryData().value(),
        result
    );
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
    detail::kernelComputeF1AndSources(
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

    // The kernel writes only F1_'s internal cells. calcDiffusivities() interpolates F1_ to the
    // faces, and the surface-interpolation boundary kernel reads F1_'s boundary values
    // (surfF1.boundary = w * F1.boundary). Without this extrapolation those boundary values are
    // uninitialised pool memory, poisoning DkEff/DomegaEff on every boundary face. Extrapolate
    // F1 (owner-cell value) to the boundary so the face blending is well-defined there.
    F1_.correctBoundaryConditions();
}

void KOmegaSST::correctNutInternal(
    const nnfvcc::VolumeField<scalar>& k,
    const nnfvcc::VolumeField<scalar>& omega,
    nnfvcc::VolumeField<scalar>& nut
) const
{
    detail::kernelCorrectNutInternal(
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
    detail::kernelCorrectNutInternal(
        exec_,
        k.boundaryData().value(),
        omega.boundaryData().value(),
        wallDist_.boundaryData().value(),
        nu_.boundaryData().value(),
        gradU_.boundaryData().value(),
        nut.boundaryData().value(),
        coeffs_.betaStar,
        coeffs_.a1,
        coeffs_.b1
    );
}

void KOmegaSST::calcDiffusivities(const nnfvcc::VolumeField<scalar>& nut)
{
    surfInterp_.interpolate(nut, surfNut_);
    surfInterp_.interpolate(F1_, surfF1_);

    detail::kernelCalcDiffusivities(
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
    detail::kernelCalcDiffusivities(
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

void KOmegaSSTModel::updateGradU(const nnfvcc::VolumeField<Vec3>& U) { model_.updateGradU(U); }

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
