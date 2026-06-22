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

namespace
{

// ---------------------------------------------------------------------------
// Robustness guards. These are overflow/edge-case limiters, NOT physics: each
// bound is set far outside any converged-solution value, so a healthy run never
// touches them. They exist only to stop a localised transient (a tiny omega, a
// degenerate wall y, or a runaway nut) from producing an Inf/NaN that FOAM's
// FOAM_SIGFPE trap turns into a hard crash mid-solve.
// ---------------------------------------------------------------------------
// Lower floor on omega used inside divisions (matches the post-solve bound()).
inline constexpr scalar KOSST_OMEGA_MIN = scalar(1e-10);
// Upper cap on the eddy viscosity nut. Physical air nut is O(1e-3..1e0); 1e6 is
// ~10 orders above that, so it only ever clips a diverging transient.
inline constexpr scalar KOSST_NUT_MAX = scalar(1e6);
// Upper cap on the pinned wall omega. omega_vis = 6 nu/(beta1 y^2) is large on
// fine meshes (1e6..1e9) but finite; this guards a degenerate near-zero y. Single-sourced from
// the omegaWallFunction BC so the BC's wall FACE value and this CELL pin use the IDENTICAL clamp.
inline constexpr scalar KOSST_OMEGA_WALL_MAX =
    NeoN::finiteVolume::cellCentred::volumeBoundary::detail::OMEGA_WF_OMEGA_MAX;

// Diagnostic that mirrors OpenFOAM's Foam::bound() (finiteVolume/lnInclude/bound.C): compute the
// GLOBAL min/max/average of a field and, ONLY when it dipped below `lowerBound` (i.e. the caller's
// max(field, lowerBound) clip actually does something), print the upstream-identical line
//   "bounding <name>, min: .. max: .. average: .."
// so neoFOAM and simpleFoam logs line up. This reports; the caller still performs the clip.
// Returns the GLOBAL min so the caller can decide whether the smoother bound() repair below is
// needed (OpenFOAM guards the repair on minVsf < lowerBound).
scalar reportBounding(
    const NeoN::Executor& exec,
    const nnfvcc::VolumeField<scalar>& field,
    const std::string& name,
    scalar lowerBound,
    bool isDistributed
)
{
    const auto view = field.internalVector().view();
    const localIdx n = static_cast<localIdx>(field.internalVector().size());

    scalar localMin = 0, localMax = 0, localSum = 0;
    Kokkos::Min<scalar> rMin(localMin);
    Kokkos::Max<scalar> rMax(localMax);
    NeoN::parallelReduce(
        exec,
        {0, n},
        NEON_LAMBDA(const localIdx i, scalar& s) {
            if (view[i] < s) s = view[i];
        },
        rMin
    );
    NeoN::parallelReduce(
        exec,
        {0, n},
        NEON_LAMBDA(const localIdx i, scalar& s) {
            if (view[i] > s) s = view[i];
        },
        rMax
    );
    NeoN::parallelReduce(
        exec,
        {0, n},
        NEON_LAMBDA(const localIdx i, scalar& s) { s += view[i]; },
        localSum
    );

    scalar gMin = localMin, gMax = localMax, gSum = localSum, gCount = static_cast<scalar>(n);
#ifdef NF_WITH_MPI_SUPPORT
    if (isDistributed)
    {
        NeoN::mpi::Environment env;
        MPI_Allreduce(MPI_IN_PLACE, &gMin, 1, NeoN::mpi::getType<scalar>(), MPI_MIN, env.comm());
        MPI_Allreduce(MPI_IN_PLACE, &gMax, 1, NeoN::mpi::getType<scalar>(), MPI_MAX, env.comm());
        scalar sumCount[2] = {gSum, gCount};
        MPI_Allreduce(MPI_IN_PLACE, sumCount, 2, NeoN::mpi::getType<scalar>(), MPI_SUM, env.comm());
        gSum = sumCount[0];
        gCount = sumCount[1];
    }
#else
    (void)isDistributed;
#endif

    if (gMin < lowerBound)
    {
        NeoN::Logging::info(
            "bounding {}, min: {} max: {} average: {}",
            name,
            gMin,
            gMax,
            gSum / gCount
        );
    }

    return gMin;
}

// Mirror of Foam::bound(vsf, lowerBound) (OpenFOAM finiteVolume/lnInclude/bound.C): a LOWER bound
// whose repair of out-of-range cells is the magSf-weighted neighbourhood average
// fvc::average(max(vsf, lb)) — NOT a hard clip to the floor. Cells that actually went <= 0 are
// replaced by that average; cells that are merely small-but-positive are clipped to lb. This is
// what keeps upstream kOmegaSST stable WITHOUT an omega ceiling: a hard floor (the previous NeoN
// behaviour) leaves the negative omega spikes that amplify, step over step, into the omega
// blow-up / SIGFPE. The caller guards on gMin < lb so this only runs on the steps that need it.
//
// fvc::average(vtf) == surfaceSum(magSf * linearInterpolate(vtf)) / surfaceSum(magSf), i.e. for
// each cell the area-weighted mean of its surrounding face values (owner AND neighbour for every
// internal face — a sign-flip-free scatter, unlike surfaceIntegrate/div). Distribution-safe: the
// floored boundary copies vsf's boundary, which already holds processor-halo neighbour values, so
// the interpolated face values (and hence the average) are consistent across ranks.
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
)
{
    const auto nCells = static_cast<localIdx>(mesh.nCells());
    const auto nInternalFaces = mesh.nInternalFaces();

    const auto magSfV = mesh.faceAreas().view();
    const auto ownersV = mesh.faceOwners().view();
    const auto neighsV = mesh.faceNeighbors().view();
    const auto bOwnersV = mesh.boundaryMesh().faceOwners().view();
    const auto bMagSfV = mesh.boundaryMesh().faceAreas().view();
    const auto nBoundaryFaces = static_cast<localIdx>(bOwnersV.size());

    // 1. floored = max(field, lowerBound) on internal cells and on boundary faces (the latter so
    //    the boundary term of fvc::average is the floored, processor-consistent value).
    {
        auto fldI = field.internalVector().view();
        auto floI = floored.internalVector().view();
        auto fldB = field.boundaryData().value().view();
        auto floB = floored.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx i) { floI[i] = Kokkos::max(fldI[i], lowerBound); },
            "kOmegaSST::boundFloorInternal"
        );
        NeoN::parallelFor(
            exec,
            {0, nBoundaryFaces},
            NEON_LAMBDA(const localIdx i) { floB[i] = Kokkos::max(fldB[i], lowerBound); },
            "kOmegaSST::boundFloorBoundary"
        );
    }

    // 2. linearInterpolate(floored) -> faces (internal + boundary, incl. processor halo).
    surfInterp.interpolate(floored, surfFloored);

    // 3. fvc::average denominator surfaceSum(magSf). Mesh-static, so build it once and cache.
    if (!sumFaceAreaBuilt)
    {
        NeoN::fill(sumFaceArea, scalar(0));
        auto sfa = sumFaceArea.view();
        NeoN::parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx f) {
                Kokkos::atomic_add(&sfa[ownersV[f]], magSfV[f]);
                Kokkos::atomic_add(&sfa[neighsV[f]], magSfV[f]);
            },
            "kOmegaSST::boundSumMagSfInternal"
        );
        NeoN::parallelFor(
            exec,
            {0, nBoundaryFaces},
            NEON_LAMBDA(const localIdx bf) { Kokkos::atomic_add(&sfa[bOwnersV[bf]], bMagSfV[bf]); },
            "kOmegaSST::boundSumMagSfBoundary"
        );
        sumFaceAreaBuilt = true;
    }

    // 4. fvc::average numerator surfaceSum(magSf * floored_face).
    NeoN::Vector<scalar> avgNum(exec, nCells, scalar(0));
    {
        auto num = avgNum.view();
        const auto sfI = surfFloored.internalVector().view();
        const auto sfB = surfFloored.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx f) {
                const scalar contrib = magSfV[f] * sfI[f];
                Kokkos::atomic_add(&num[ownersV[f]], contrib);
                Kokkos::atomic_add(&num[neighsV[f]], contrib);
            },
            "kOmegaSST::boundAvgNumInternal"
        );
        NeoN::parallelFor(
            exec,
            {0, nBoundaryFaces},
            NEON_LAMBDA(const localIdx bf) {
                Kokkos::atomic_add(&num[bOwnersV[bf]], bMagSfV[bf] * sfB[bf]);
            },
            "kOmegaSST::boundAvgNumBoundary"
        );
    }

    // 5. Repair: cells that went <= 0 take the neighbourhood average; everything else is floored.
    //    Mirrors  max( max(vsf, avg * pos0(-vsf)), lowerBound )  with avg >= lowerBound by
    //    construction (it is an average of floored values).
    {
        auto fldI = field.internalVector().view();
        const auto num = avgNum.view();
        const auto sfa = sumFaceArea.view();
        NeoN::parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx i) {
                const scalar avg = num[i] / sfa[i];
                const scalar repaired = (fldI[i] <= scalar(0)) ? avg : fldI[i];
                fldI[i] = Kokkos::max(repaired, lowerBound);
            },
            "kOmegaSST::boundSmoothRepair"
        );
    }
}

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
    // Local copies of the namespace guards: an anonymous-namespace constexpr is
    // not addressable in device code, so capture them by value (like rootVSmall).
    const scalar omegaMin = KOSST_OMEGA_MIN;
    const scalar nutMax = KOSST_NUT_MAX;

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(kVec.size())},
        NEON_LAMBDA(const localIdx i) {
            const scalar k_i = kV[i];
            const scalar omega_i = omegaV[i];
            const scalar nu_i = nuV[i];
            const scalar y_i = Kokkos::max(wallDistV[i], rootVSmall);
            const scalar y2_i = y_i * y_i;

            // Floored omega/k used in every division and limiter below (overflow
            // guard; a healthy field is far above the floor so these are no-ops).
            const scalar omegaSafe = Kokkos::max(omega_i, omegaMin);
            const scalar kSafe = Kokkos::max(k_i, scalar(0));

            // ----- CDkOmega (cross-diffusion, clamped for F1 stability) -----
            const scalar dotGradKOmega = gradKV[i][0] * gradOmegaV[i][0]
                                       + gradKV[i][1] * gradOmegaV[i][1]
                                       + gradKV[i][2] * gradOmegaV[i][2];

            const scalar CDkOmegaPlus =
                Kokkos::max(scalar(2) * alphaOmega2 * dotGradKOmega / omegaSafe, scalar(1e-10));

            // ----- F1 blending (inner ↔ outer) -----
            const scalar sqrtK = Kokkos::sqrt(Kokkos::max(k_i, scalar(0)));

            const scalar arg1 = Kokkos::min(
                Kokkos::min(
                    Kokkos::max(
                        sqrtK / (betaStar * omegaSafe * y_i),
                        scalar(500) * nu_i / (y2_i * omegaSafe)
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
                    scalar(2) * sqrtK / (betaStar * omegaSafe * y_i),
                    scalar(500) * nu_i / (y2_i * omegaSafe)
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
            // OLD nut for G (consistent with OF sequence), clamped to [0, NUT_MAX].
            const scalar nut_i = Kokkos::min(Kokkos::max(nutV[i], scalar(0)), nutMax);

            // ----- Blended coefficients -----
            const scalar gamma_i = F1_i * (gamma1 - gamma2) + gamma2;
            const scalar beta_i = F1_i * (beta1 - beta2) + beta2;

            // ----- k equation sources -----
            // G = nut * GbyNu0, Pk = min(G, c1*betaStar*k*omega)
            const scalar G_i = nut_i * GbyNu0_i;
            PkV[i] = Kokkos::min(G_i, c1 * betaStar * kSafe * omegaSafe);

            // betaStar*omega as implicit destruction for k
            spKV[i] = betaStar * omegaSafe;

            // ----- omega equation sources -----
            // Bounded GbyNu for omega production
            const scalar GbyNuBound_i = Kokkos::min(
                GbyNu0_i,
                (c1 / a1) * betaStar * omegaSafe * Kokkos::max(a1 * omegaSafe, b1 * F2_i * sqrtS2)
            );
            omegaSourceV[i] = gamma_i * GbyNuBound_i;

            // beta*omega as implicit destruction for omega (base)
            spOmegaV[i] = beta_i * omegaSafe;

            // Cross-diffusion: (1-F1)*CDkOmega (actual, may be negative)
            const scalar CDkOmegaActual = scalar(2) * alphaOmega2 * dotGradKOmega / omegaSafe;
            const scalar crossSource = (scalar(1) - F1_i) * CDkOmegaActual;

            // Positive cross-source → explicit; negative → implicit sink for stability
            omegaSourceV[i] += Kokkos::max(crossSource, scalar(0));
            spOmegaV[i] += Kokkos::max(-crossSource / omegaSafe, scalar(0));
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
    // Local copies of the namespace guards (not addressable in device code).
    const scalar omegaMin = KOSST_OMEGA_MIN;
    const scalar nutMax = KOSST_NUT_MAX;

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(kVec.size())},
        NEON_LAMBDA(const localIdx i) {
            const scalar k_i = kV[i];
            const scalar omega_i = omegaV[i];
            const scalar nu_i = nuV[i];
            const scalar y_i = Kokkos::max(wallDistV[i], rootVSmall);
            const scalar y2_i = y_i * y_i;

            // Floored omega/k for divisions and the nut denominator (overflow guard).
            const scalar omegaSafe = Kokkos::max(omega_i, omegaMin);
            const scalar kSafe = Kokkos::max(k_i, scalar(0));

            // F2
            const scalar sqrtK = Kokkos::sqrt(kSafe);
            const scalar arg2 = Kokkos::min(
                Kokkos::max(
                    scalar(2) * sqrtK / (betaStar * omegaSafe * y_i),
                    scalar(500) * nu_i / (y2_i * omegaSafe)
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

            // nut = a1*k / max(a1*omega, b1*F2*sqrt(S2)), capped to [0, NUT_MAX].
            // The denominator already has a positive floor via a1*omegaSafe, so the
            // division is finite; the cap stops a transient k spike (or vanishing
            // strain+omega) from producing an Inf that the FP trap would catch.
            const scalar nutRaw = a1 * kSafe / Kokkos::max(a1 * omegaSafe, b1 * F2_i * sqrtS2);
            nutV_[i] = Kokkos::min(nutRaw, nutMax);
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

// Populate nearWallDist's boundary face values from wallDist's owner-cell internal value.
// Kept as a free function (not a constructor body / member) because NVCC forbids extended
// __host__ __device__ lambdas (NEON_LAMBDA) inside functions whose address can't be taken
// (constructors) or inside private/protected members.
void initNearWallDistBoundary(
    const NeoN::Executor& exec,
    const nnfvcc::VolumeField<scalar>& wallDist,
    const NeoN::UnstructuredMesh& mesh,
    nnfvcc::VolumeField<scalar>& nearWallDist
)
{
    const auto wdInternal = wallDist.internalVector().view();
    const auto faceOwners = mesh.boundaryMesh().faceOwners().view();
    auto nwdBoundary = nearWallDist.boundaryData().value().view();
    const auto nBoundaryFaces = static_cast<NeoN::localIdx>(nwdBoundary.size());
    NeoN::parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const NeoN::localIdx i) { nwdBoundary[i] = wdInternal[faceOwners[i]]; },
        "kOmegaSST::initNearWallDistBoundary"
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
    initNearWallDistBoundary(exec_, wallDist_, mesh_, nearWallDist_);
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
        const scalar gMin = reportBounding(
            exec_,
            omega,
            "omega",
            KOSST_OMEGA_MIN,
            mesh_.boundaryMesh().isDistributed()
        );
        if (gMin < KOSST_OMEGA_MIN)
        {
            boundLowerSmoothRepair(
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
            reportBounding(exec_, k, "k", scalar(0), mesh_.boundaryMesh().isDistributed());
        if (gMin < scalar(0))
        {
            boundLowerSmoothRepair(
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
    kernelCorrectNutInternal(
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
