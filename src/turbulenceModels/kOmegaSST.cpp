// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/turbulenceModels/kOmegaSST.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "NeoFOAM/auxiliary/writers.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"
#include "NeoFOAM/auxiliary/bound.hpp"
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

// Named per-model so this model's SYCL device-kernel names stay unique across TUs.
namespace kOmegaSSTDetail
{
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
    NeoN::Vector<scalar>& f1Vec,
    NeoN::Vector<scalar>& pkVec,
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
    const NeoN::Vector<scalar>& surff1Vec,
    NeoN::Vector<scalar>& nuEffVec,
    NeoN::Vector<scalar>& dkEffVec,
    NeoN::Vector<scalar>& domegaEffVec,
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
} // namespace kOmegaSSTDetail

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
    , nearWallDistTmp_(
          exec,
          "nearWallDist",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , surfNuTmp_(
          exec,
          "surfNu",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , gradUTmp_(
          exec,
          "gradU",
          mesh,
          fvcc::createCalculatedProcBCs<nnfvcc::VolumeBoundary<Tensor>>(mesh)
      )
    , gradKTmp_(exec, "gradK", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<Vec3>>(mesh))
    , gradOmegaTmp_(
          exec,
          "gradOmega",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<Vec3>>(mesh)
      )
    , F1Tmp_(exec, "F1", mesh, fvcc::createExtrapolatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh))
    , PkTmp_(exec, "Pk", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh))
    , spKTmp_(exec, "spK", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh))
    , omegaSourceTmp_(
          exec,
          "omegaSource",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , spOmegaTmp_(
          exec,
          "spOmega",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , surfNutTmp_(
          exec,
          "surfNut",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , surfF1Tmp_(
          exec,
          "surfF1",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , nuEffTmp_(
          exec,
          "nuEff",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , dkEffFTmp_(
          exec,
          "DkEff",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , domegaEffFTmp_(
          exec,
          "DomegaEff",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , gradOp_(exec, mesh)
    , gradUOp_(nnfvcc::GradOperatorFactory<Vec3>::create(
          exec,
          mesh,
          NeoN::TokenList({std::string("Gauss"), std::string("linear")})
      ))
    , surfInterp_(exec, mesh, NeoN::TokenList({std::string("linear")}))
    , coeffs_()
    , omegaWallValueTmp_(exec, static_cast<NeoN::localIdx>(mesh.nCells()), scalar(0))
    , omegaWallMaskTmp_(exec, static_cast<NeoN::localIdx>(mesh.nCells()), scalar(0))
    , cornerWeightTmp_(exec, static_cast<NeoN::localIdx>(mesh.nCells()), scalar(0))
{
    surfInterp_.interpolate(nu_, surfNuTmp_);

    // NVCC forbids NEON_LAMBDA in a constructor body — use a free function.
    kOmegaSSTDetail::initNearWallDistBoundary(exec_, wallDist_, mesh_, nearWallDistTmp_);
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
    reserveScratch(); // re-grow per-step scratch released by a previous correct()/validate()

    gradUOp_->gradTensor(U, gradUTmp_, NeoN::dsl::Coeff {});
    gradUTmp_.correctBoundaryConditions();
    gradOp_.grad(k, gradKTmp_);
    gradOp_.grad(omega, gradOmegaTmp_);

    computeF1AndSources(k, omega, nut, gradKTmp_, gradOmegaTmp_, gradUTmp_);

    correctNutInternal(k, omega, nut);
    fvcc::BoundaryContext ctx;
    ctx.insert("U", U);
    ctx.insert("k", k);
    ctx.insert("nu", nu_);
    ctx.insert("nearWallDist", nearWallDistTmp_);
    nut.correctBoundaryConditions(ctx);

    calcDiffusivities(nut);

    releaseScratch(); // free per-step scratch until the next correct()
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
    reserveScratch();

    // Compute gradients once — reused for production, CDkOmega, and correctNut
    gradUOp_->gradTensor(U, gradUTmp_, NeoN::dsl::Coeff {});
    gradUTmp_.correctBoundaryConditions();
    gradOp_.grad(k, gradKTmp_);
    gradOp_.grad(omega, gradOmegaTmp_);

    fvcc::BoundaryContext ctx;
    ctx.insert("U", U);
    ctx.insert("phi", phi);
    ctx.insert("k", k);
    ctx.insert("nu", nu_);
    ctx.insert("nearWallDist", nearWallDistTmp_);

    // Compute F1, sources, and sp coefficients using OLD nut
    computeF1AndSources(k, omega, nut, gradKTmp_, gradOmegaTmp_, gradUTmp_);

    freeVecs(gradKTmp_.internalVector(), gradOmegaTmp_.internalVector());
    calcDiffusivities(nut);

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
        const auto yBoundaryV = nearWallDistTmp_.boundaryData().value().view();
        const auto kInternalV = k.internalVector().view();
        auto pkInternalV = PkTmp_.internalVector().view();
        const auto nCells = static_cast<NeoN::localIdx>(mesh_.nCells());

        // One-time: build cornerWeightTmp_[c] = 1/(number of omegaWallFunction faces touching cell
        // c), 0 for non-wall cells. The wall topology is static so this is computed once. Mirrors
        // OpenFOAM omegaWallFunction::createAveragingWeights (omega.C:71-126).
        if (!cornerWeightsBuilt_)
        {
            NeoN::fill(cornerWeightTmp_, scalar(0));
            auto cwBuild = cornerWeightTmp_.view();
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
        const auto cornerWeightV = cornerWeightTmp_.view();

        // Per-step reset of the accumulators: the pin mask, the pinned value (now ACCUMULATED, so
        // it must start at zero), and PkTmp_ at the wall cells (computeF1AndSources wrote the bulk
        // production there; the wall log-law production replaces it via the weighted sum below).
        NeoN::fill(omegaWallMaskTmp_, scalar(0));
        NeoN::fill(omegaWallValueTmp_, scalar(0));
        auto omegaWallValueV = omegaWallValueTmp_.view();
        auto omegaWallMaskV = omegaWallMaskTmp_.view();
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

                    const scalar ySafe = Kokkos::max(y, scalar(1e-30));
                    const scalar wVis = scalar(6) * nuw / (beta1 * ySafe * ySafe);
                    const scalar wLog = Kokkos::sqrt(kc) / (Cmu25 * kappa_ * ySafe);
                    const scalar wOmega =
                        Kokkos::min(Kokkos::sqrt(wVis * wVis + wLog * wLog), omegaWallMax);

                    Kokkos::atomic_add(&pkInternalV[owner], cw * gWall);
                    Kokkos::atomic_add(&omegaWallValueV[owner], cw * wOmega);
                    omegaWallMaskV[owner] = scalar(1);
                },
                "kOmegaSST::omegaWFGFeedback"
            );
        }
    }

    // ----- omega equation -----
    PDESolver<scalar> omegaEqn(
        dsl::imp::ddt(omega) + dsl::imp::div(phi, omega)
            - dsl::imp::laplacian(domegaEffFTmp_, omega) + dsl::imp::source(spOmegaTmp_, omega)
            - dsl::exp::source(omegaSourceTmp_),
        omega,
        rt
    );
    // Hard-pin the near-wall cells to the blended wall omega built above (the missing
    // omegaWallFunction setValues equivalent). Skipped when no omegaWallFunction patch exists.
    if (anyOmegaWF)
    {
        omegaEqn.setConstraints(omegaWallMaskTmp_, omegaWallValueTmp_);
    }
    omegaEqn.solve();

    bound(omega, KOSST_OMEGA_MIN, boundCache_);
    omega.correctBoundaryConditions(ctx);

    freeVecs(
        omegaSourceTmp_.internalVector(),
        spOmegaTmp_.internalVector(),
        domegaEffFTmp_.internalVector(),
        omegaWallValueTmp_,
        omegaWallMaskTmp_
    );

    {
        const scalar betaStar = coeffs_.betaStar;
        auto omegaView = omega.internalVector().view();
        auto spKView = spKTmp_.internalVector().view();
        NeoN::parallelFor(
            exec_,
            {0, static_cast<localIdx>(omega.internalVector().size())},
            NEON_LAMBDA(const localIdx i) { spKView[i] = betaStar * omegaView[i]; },
            "kOmegaSST::updateSpKFromNewOmega"
        );
    }

    // ----- k equation -----
    PDESolver<scalar> kEqn(
        dsl::imp::ddt(k) + dsl::imp::div(phi, k) - dsl::imp::laplacian(dkEffFTmp_, k)
            + dsl::imp::source(spKTmp_, k) - dsl::exp::source(PkTmp_),
        k,
        rt
    );
    kEqn.solve();

    bound(k, kMin_, boundCache_);
    k.correctBoundaryConditions(ctx);

    freeVecs(PkTmp_.internalVector(), spKTmp_.internalVector(), dkEffFTmp_.internalVector());
    correctNutInternal(k, omega, nut);
    nut.correctBoundaryConditions(ctx);

    // Refresh surface diffusivities with new nut
    calcDiffusivities(nut);

    releaseScratch();
}

nnfvcc::SurfaceField<scalar>& KOmegaSST::nuEff() { return nuEffTmp_; }

nnfvcc::SurfaceField<scalar>& KOmegaSST::dkEff() { return dkEffFTmp_; }

nnfvcc::SurfaceField<scalar>& KOmegaSST::domegaEff() { return domegaEffFTmp_; }

const nnfvcc::VolumeField<Tensor>& KOmegaSST::gradU() const { return gradUTmp_; }

void KOmegaSST::updateGradU(const nnfvcc::VolumeField<Vec3>& U)
{
    gradUOp_->gradTensor(U, gradUTmp_, NeoN::dsl::Coeff {});
    gradUTmp_.correctBoundaryConditions();
}

NeoN::Vector<SymmTensor> KOmegaSST::devRhoReff() const
{
    const localIdx nBF = static_cast<localIdx>(gradUTmp_.boundaryData().value().size());
    NeoN::Vector<SymmTensor> result(exec_, nBF, NeoN::zero<SymmTensor>());
    kOmegaSSTDetail::kernelDevRhoReff(
        exec_,
        gradUTmp_.boundaryData().value(),
        nuEffTmp_.boundaryData().value(),
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
    kOmegaSSTDetail::kernelComputeF1AndSources(
        exec_,
        k.internalVector(),
        omega.internalVector(),
        nut.internalVector(),
        nu_.internalVector(),
        wallDist_.internalVector(),
        gradK.internalVector(),
        gradOmega.internalVector(),
        gradU.internalVector(),
        F1Tmp_.internalVector(),
        PkTmp_.internalVector(),
        spKTmp_.internalVector(),
        omegaSourceTmp_.internalVector(),
        spOmegaTmp_.internalVector(),
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

    // Extrapolate F1 to boundary faces — surface interpolation reads these values.
    F1Tmp_.correctBoundaryConditions();
}

void KOmegaSST::correctNutInternal(
    const nnfvcc::VolumeField<scalar>& k,
    const nnfvcc::VolumeField<scalar>& omega,
    nnfvcc::VolumeField<scalar>& nut
) const
{
    kOmegaSSTDetail::kernelCorrectNutInternal(
        exec_,
        k.internalVector(),
        omega.internalVector(),
        wallDist_.internalVector(),
        nu_.internalVector(),
        gradUTmp_.internalVector(),
        nut.internalVector(),
        coeffs_.betaStar,
        coeffs_.a1,
        coeffs_.b1
    );
    kOmegaSSTDetail::kernelCorrectNutInternal(
        exec_,
        k.boundaryData().value(),
        omega.boundaryData().value(),
        wallDist_.boundaryData().value(),
        nu_.boundaryData().value(),
        gradUTmp_.boundaryData().value(),
        nut.boundaryData().value(),
        coeffs_.betaStar,
        coeffs_.a1,
        coeffs_.b1
    );
}

void KOmegaSST::calcDiffusivities(const nnfvcc::VolumeField<scalar>& nut)
{
    const auto nsf = surfNuTmp_.internalVector().size();
    auto ensure = [](auto& v, NeoN::localIdx n)
    {
        if (v.size() != n) v.resize(n);
    };
    ensure(surfNutTmp_.internalVector(), nsf);
    ensure(surfF1Tmp_.internalVector(), nsf);
    ensure(nuEffTmp_.internalVector(), nsf);
    ensure(dkEffFTmp_.internalVector(), nsf);
    ensure(domegaEffFTmp_.internalVector(), nsf);

    surfInterp_.interpolate(nut, surfNutTmp_);
    surfInterp_.interpolate(F1Tmp_, surfF1Tmp_);

    kOmegaSSTDetail::kernelCalcDiffusivities(
        exec_,
        surfNuTmp_.internalVector(),
        surfNutTmp_.internalVector(),
        surfF1Tmp_.internalVector(),
        nuEffTmp_.internalVector(),
        dkEffFTmp_.internalVector(),
        domegaEffFTmp_.internalVector(),
        coeffs_.alphaK1,
        coeffs_.alphaK2,
        coeffs_.alphaOmega1,
        coeffs_.alphaOmega2,
        "kOmegaSST::calcDiffusivities::internal"
    );
    kOmegaSSTDetail::kernelCalcDiffusivities(
        exec_,
        surfNuTmp_.boundaryData().value(),
        surfNutTmp_.boundaryData().value(),
        surfF1Tmp_.boundaryData().value(),
        nuEffTmp_.boundaryData().value(),
        dkEffFTmp_.boundaryData().value(),
        domegaEffFTmp_.boundaryData().value(),
        coeffs_.alphaK1,
        coeffs_.alphaK2,
        coeffs_.alphaOmega1,
        coeffs_.alphaOmega2,
        "kOmegaSST::calcDiffusivities::boundary"
    );

    freeVecs(surfNutTmp_.internalVector(), surfF1Tmp_.internalVector());
    NeoN::fence(exec_);
}

void KOmegaSST::reserveScratch()
{
    const auto nc = static_cast<NeoN::localIdx>(mesh_.nCells());
    auto ensure = [](auto& vec, NeoN::localIdx n)
    {
        if (vec.size() != n) vec.resize(n);
    };
    ensure(gradKTmp_.internalVector(), nc);
    ensure(gradOmegaTmp_.internalVector(), nc);
    ensure(F1Tmp_.internalVector(), nc);
    ensure(PkTmp_.internalVector(), nc);
    ensure(spKTmp_.internalVector(), nc);
    ensure(omegaSourceTmp_.internalVector(), nc);
    ensure(spOmegaTmp_.internalVector(), nc);
    // plain per-cell Vectors, rebuilt (filled) every correct()
    ensure(omegaWallValueTmp_, nc);
    ensure(omegaWallMaskTmp_, nc);
}

bool KOmegaSST::devicePoolActive()
{
    static const bool active = []
    {
#if NF_WITH_UMPIRE && defined(KOKKOS_ENABLE_CUDA)
        try
        {
            (void)NeoN::UmpireMempoolHandler::getUmpirePool(NeoN::MemorySpace::GPU);
            return true;
        }
        catch (...)
        {
            return false;
        }
#else
        return false;
#endif
    }();
    return active;
}

void KOmegaSST::releaseScratch()
{
    freeVecs(
        gradKTmp_.internalVector(),
        gradOmegaTmp_.internalVector(),
        F1Tmp_.internalVector(),
        PkTmp_.internalVector(),
        spKTmp_.internalVector(),
        omegaSourceTmp_.internalVector(),
        spOmegaTmp_.internalVector(),
        surfNutTmp_.internalVector(),
        surfF1Tmp_.internalVector(),
        dkEffFTmp_.internalVector(),
        domegaEffFTmp_.internalVector(),
        omegaWallValueTmp_,
        omegaWallMaskTmp_
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
    // grad(U) honours the configured gradSchemes; the operator is shared with the
    // solver's other grad(U) call sites via RunTime's cache.
    model_.setGradUOperator(gradSchemePtr(rt, "grad(U)"));

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
