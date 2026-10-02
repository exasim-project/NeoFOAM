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

namespace NeoFOAM::kOmegaSSTDetail
{

inline constexpr scalar KOSST_OMEGA_MIN = scalar(1e-10);
inline constexpr scalar KOSST_NUT_MAX = scalar(1e6);
inline constexpr scalar KOSST_OMEGA_WALL_MAX =
    NeoN::finiteVolume::cellCentred::volumeBoundary::detail::OMEGA_WF_OMEGA_MAX;

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
         f1V,
         pkV,
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
                f1Vec,
                pkVec,
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
            const scalar kI = kV[i];
            const scalar omegaI = omegaV[i];
            const scalar nuI = nuV[i];
            const scalar yI = Kokkos::max(wallDistV[i], rootVSmall);
            const scalar y2I = yI * yI;

            const scalar omegaSafe = Kokkos::max(omegaI, omegaMin);
            const scalar kSafe = Kokkos::max(kI, scalar(0));

            const scalar dotGradKOmega = gradKV[i][0] * gradOmegaV[i][0]
                                       + gradKV[i][1] * gradOmegaV[i][1]
                                       + gradKV[i][2] * gradOmegaV[i][2];

            const scalar cDkOmegaPlus =
                Kokkos::max(scalar(2) * alphaOmega2 * dotGradKOmega / omegaSafe, scalar(1e-10));

            const scalar sqrtK = Kokkos::sqrt(Kokkos::max(kI, scalar(0)));

            const scalar arg1 = Kokkos::min(
                Kokkos::min(
                    Kokkos::max(
                        sqrtK / (betaStar * omegaSafe * yI),
                        scalar(500) * nuI / (y2I * omegaSafe)
                    ),
                    scalar(4) * alphaOmega2 * kI / (cDkOmegaPlus * y2I)
                ),
                scalar(10)
            );
            const scalar arg14 = arg1 * arg1 * arg1 * arg1;
            f1V[i] = Kokkos::tanh(arg14);
            const scalar f1I = f1V[i];

            const scalar arg2 = Kokkos::min(
                Kokkos::max(
                    scalar(2) * sqrtK / (betaStar * omegaSafe * yI),
                    scalar(500) * nuI / (y2I * omegaSafe)
                ),
                scalar(100)
            );
            const scalar f2I = Kokkos::tanh(arg2 * arg2);

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

            const scalar s2I = normSq + dotTrans;
            const scalar gbyNu0I = s2I - (scalar(2) / scalar(3)) * divU * divU;

            const scalar sqrtS2 = Kokkos::sqrt(Kokkos::max(s2I, scalar(0)));
            const scalar nutI = Kokkos::min(Kokkos::max(nutV[i], scalar(0)), nutMax);

            const scalar gammaI = f1I * (gamma1 - gamma2) + gamma2;
            const scalar betaI = f1I * (beta1 - beta2) + beta2;

            const scalar gI = nutI * gbyNu0I;
            pkV[i] = Kokkos::min(gI, c1 * betaStar * kSafe * omegaSafe);

            spKV[i] = betaStar * omegaSafe;

            const scalar gbyNuBoundI = Kokkos::min(
                gbyNu0I,
                (c1 / a1) * betaStar * omegaSafe * Kokkos::max(a1 * omegaSafe, b1 * f2I * sqrtS2)
            );
            omegaSourceV[i] = gammaI * gbyNuBoundI;

            spOmegaV[i] = betaI * omegaSafe;

            const scalar CDkOmegaActual = scalar(2) * alphaOmega2 * dotGradKOmega / omegaSafe;
            const scalar crossSource = (scalar(1) - f1I) * CDkOmegaActual;

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
            const scalar kI = kV[i];
            const scalar omegaI = omegaV[i];
            const scalar nuI = nuV[i];
            const scalar yI = Kokkos::max(wallDistV[i], rootVSmall);
            const scalar y2I = yI * yI;

            const scalar omegaSafe = Kokkos::max(omegaI, omegaMin);
            const scalar kSafe = Kokkos::max(kI, scalar(0));

            const scalar sqrtK = Kokkos::sqrt(kSafe);
            const scalar arg2 = Kokkos::min(
                Kokkos::max(
                    scalar(2) * sqrtK / (betaStar * omegaSafe * yI),
                    scalar(500) * nuI / (y2I * omegaSafe)
                ),
                scalar(100)
            );
            const scalar f2I = Kokkos::tanh(arg2 * arg2);

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
            const scalar s2I = normSq + dotTrans;
            const scalar sqrtS2 = Kokkos::sqrt(Kokkos::max(s2I, scalar(0)));

            const scalar nutRaw = a1 * kSafe / Kokkos::max(a1 * omegaSafe, b1 * f2I * sqrtS2);
            nutV_[i] = Kokkos::min(nutRaw, nutMax);
        },
        "kOmegaSST::correctNutInternal"
    );
}

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
)
{
    const auto [nuF, nutF, f1F, nuEffF, dkF, domF] =
        NeoN::views(surfNuVec, surfNutVec, surff1Vec, nuEffVec, dkEffVec, domegaEffVec);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(nuEffVec.size())},
        NEON_LAMBDA(const localIdx f) {
            const scalar alphaK = f1F[f] * (alphaK1 - alphaK2) + alphaK2;
            const scalar alphaOmega = f1F[f] * (alphaOmega1 - alphaOmega2) + alphaOmega2;

            nuEffF[f] = nuF[f] + nutF[f];
            dkF[f] = alphaK * nutF[f] + nuF[f];
            domF[f] = alphaOmega * nutF[f] + nuF[f];
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

} // namespace NeoFOAM::kOmegaSSTDetail
