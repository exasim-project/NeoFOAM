// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

// Kernel functions for kOmegaSST split into a separate TU so each file stays
// within Intel PVC's per-TU AOT compilation limit (~2 passes).

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/turbulenceModels/kOmegaSST.hpp"
#include "NeoFOAM/fvcc/boundary/volume/omegaWallFunction.hpp"

using NeoN::localIdx;
using NeoN::Tensor;
using NeoN::SymmTensor;

namespace NeoFOAM::detail
{

inline constexpr scalar KOSST_OMEGA_MIN = scalar(1e-10);
inline constexpr scalar KOSST_NUT_MAX = scalar(1e6);
inline constexpr scalar KOSST_OMEGA_WALL_MAX =
    NeoN::finiteVolume::cellCentred::volumeBoundary::detail::OMEGA_WF_OMEGA_MAX;

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

    surfInterp.interpolate(floored, surfFloored);

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

            const scalar omegaSafe = Kokkos::max(omega_i, omegaMin);
            const scalar kSafe = Kokkos::max(k_i, scalar(0));

            const scalar dotGradKOmega = gradKV[i][0] * gradOmegaV[i][0]
                                       + gradKV[i][1] * gradOmegaV[i][1]
                                       + gradKV[i][2] * gradOmegaV[i][2];

            const scalar CDkOmegaPlus =
                Kokkos::max(scalar(2) * alphaOmega2 * dotGradKOmega / omegaSafe, scalar(1e-10));

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

            const scalar arg2 = Kokkos::min(
                Kokkos::max(
                    scalar(2) * sqrtK / (betaStar * omegaSafe * y_i),
                    scalar(500) * nu_i / (y2_i * omegaSafe)
                ),
                scalar(100)
            );
            const scalar F2_i = Kokkos::tanh(arg2 * arg2);

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

            const scalar sqrtS2 = Kokkos::sqrt(Kokkos::max(S2_i, scalar(0)));
            const scalar nut_i = Kokkos::min(Kokkos::max(nutV[i], scalar(0)), nutMax);

            const scalar gamma_i = F1_i * (gamma1 - gamma2) + gamma2;
            const scalar beta_i = F1_i * (beta1 - beta2) + beta2;

            const scalar G_i = nut_i * GbyNu0_i;
            PkV[i] = Kokkos::min(G_i, c1 * betaStar * kSafe * omegaSafe);

            spKV[i] = betaStar * omegaSafe;

            const scalar GbyNuBound_i = Kokkos::min(
                GbyNu0_i,
                (c1 / a1) * betaStar * omegaSafe * Kokkos::max(a1 * omegaSafe, b1 * F2_i * sqrtS2)
            );
            omegaSourceV[i] = gamma_i * GbyNuBound_i;

            spOmegaV[i] = beta_i * omegaSafe;

            const scalar CDkOmegaActual = scalar(2) * alphaOmega2 * dotGradKOmega / omegaSafe;
            const scalar crossSource = (scalar(1) - F1_i) * CDkOmegaActual;

            omegaSourceV[i] += Kokkos::max(crossSource, scalar(0));
            spOmegaV[i] += Kokkos::max(-crossSource / omegaSafe, scalar(0));
        },
        "kOmegaSST::computeF1AndSources"
    );
}

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

            const scalar omegaSafe = Kokkos::max(omega_i, omegaMin);
            const scalar kSafe = Kokkos::max(k_i, scalar(0));

            const scalar sqrtK = Kokkos::sqrt(kSafe);
            const scalar arg2 = Kokkos::min(
                Kokkos::max(
                    scalar(2) * sqrtK / (betaStar * omegaSafe * y_i),
                    scalar(500) * nu_i / (y2_i * omegaSafe)
                ),
                scalar(100)
            );
            const scalar F2_i = Kokkos::tanh(arg2 * arg2);

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

            const scalar nutRaw = a1 * kSafe / Kokkos::max(a1 * omegaSafe, b1 * F2_i * sqrtS2);
            nutV_[i] = Kokkos::min(nutRaw, nutMax);
        },
        "kOmegaSST::correctNutInternal"
    );
}

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

} // namespace NeoFOAM::detail
