// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstddef>
#include <stdexcept>
#include <string>

#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include "NeoFOAM/algorithms/vof/mules.hpp"
#include "detail.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

namespace
{

using NeoN::localIdx;
using NeoN::scalar;

// OpenFOAM doubleScalar constants (doubleScalar.H).
constexpr scalar rootVSmall = 1.0e-150;
constexpr scalar small = 1.0e-15;

// fvc::surfaceIntegrate: owner += flux, neighbour -= flux, boundary owner += flux, then / V.
// NeoN's surfaceIntegrate multiplies by 1/V, which rounds differently from OpenFOAM's division.
NeoN::Vector<scalar> integrateFlux(
    const NeoN::UnstructuredMesh& mesh,
    NeoN::View<const scalar> flux,
    NeoN::View<const scalar> bFlux
)
{
    const auto exec = mesh.exec();
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    const auto V = mesh.cellVolumes().view();
    NeoN::Vector<scalar> result(exec, mesh.nCells(), 0.0);
    auto res = result.view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) {
            Kokkos::atomic_add(&res[own[f]], flux[f]);
            Kokkos::atomic_sub(&res[nei[f]], flux[f]);
        },
        "vof::integrateInternal"
    );
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { Kokkos::atomic_add(&res[bOwn[bf]], bFlux[bf]); },
        "vof::integrateBoundary"
    );
    NeoN::parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { res[c] /= V[c]; },
        "vof::integrateVolume"
    );
    return result;
}

// fvPatchField::fixesValue() as OpenFOAM answers it: NeoN marks inletOutlet as not fixing the
// value, but OpenFOAM's inletOutlet is a mixedFvPatchField, whose fixesValue() is true.
bool fixesValueInOpenFOAM(const fvcc::VolumeBoundary<scalar>& bc)
{
    return bc.attributes().fixesValue || bc.name() == "inletOutlet";
}

// Per-cell inputs of the FCT limiter: local extrema psiMaxn/psiMinn (only the neighbours and
// fixed-value boundary values, as in MULES::limiter; the cell's own value is not included) and
// the sums of the outgoing (sumPhip) and incoming (mSumPhim) correction fluxes.
struct LimiterSums
{
    NeoN::Vector<scalar> psiMaxn;
    NeoN::Vector<scalar> psiMinn;
    NeoN::Vector<scalar> sumPhip;
    NeoN::Vector<scalar> mSumPhim;
};

LimiterSums limiterSums(
    const fvcc::VolumeField<scalar>& psi,
    NeoN::View<const scalar> phiCorr,
    NeoN::View<const scalar> bPhiCorr,
    scalar psiMax,
    scalar psiMin
)
{
    const auto& mesh = psi.mesh();
    const auto exec = psi.exec();
    const auto nCells = mesh.nCells();
    LimiterSums sums {
        NeoN::Vector<scalar>(exec, nCells, psiMin),
        NeoN::Vector<scalar>(exec, nCells, psiMax),
        NeoN::Vector<scalar>(exec, nCells, 0.0),
        NeoN::Vector<scalar>(exec, nCells, 0.0)
    };
    auto psiMaxn = sums.psiMaxn.view();
    auto psiMinn = sums.psiMinn.view();
    auto sumPhip = sums.sumPhip.view();
    auto mSumPhim = sums.mSumPhim.view();

    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    const auto psiI = psi.internalVector().view();
    const auto psiB = psi.boundaryData().value().view();

    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) {
            const localIdx o = own[f];
            const localIdx n = nei[f];
            Kokkos::atomic_max(&psiMaxn[o], psiI[n]);
            Kokkos::atomic_min(&psiMinn[o], psiI[n]);
            Kokkos::atomic_max(&psiMaxn[n], psiI[o]);
            Kokkos::atomic_min(&psiMinn[n], psiI[o]);
            const scalar pc = phiCorr[f];
            if (pc > 0.0)
            {
                Kokkos::atomic_add(&sumPhip[o], pc);
                Kokkos::atomic_add(&mSumPhim[n], pc);
            }
            else
            {
                Kokkos::atomic_sub(&mSumPhim[o], pc);
                Kokkos::atomic_sub(&sumPhip[n], pc);
            }
        },
        "mules::sumsInternal"
    );

    const auto bcs = psi.boundaryConditions();
    for (std::size_t patchi = 0; patchi < bcs.size(); ++patchi)
    {
        const auto [start, end] = psi.boundaryData().range(static_cast<localIdx>(patchi));
        if (start == end)
        {
            continue;
        }
        if (bcs[patchi].name() == "processor")
        {
            throw std::runtime_error(
                "MULES: coupled (processor) patches are not supported: the local extrema "
                "would need the patch-neighbour values"
            );
        }
        if (fixesValueInOpenFOAM(bcs[patchi]))
        {
            NeoN::parallelFor(
                exec,
                {start, end},
                NEON_LAMBDA(const localIdx bf) {
                    Kokkos::atomic_max(&psiMaxn[bOwn[bf]], psiB[bf]);
                    Kokkos::atomic_min(&psiMinn[bOwn[bf]], psiB[bf]);
                },
                "mules::fixedValueExtrema"
            );
        }
    }

    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) {
            const scalar pc = bPhiCorr[bf];
            if (pc > 0.0)
            {
                Kokkos::atomic_add(&sumPhip[bOwn[bf]], pc);
            }
            else
            {
                Kokkos::atomic_sub(&mSumPhim[bOwn[bf]], pc);
            }
        },
        "mules::sumsBoundary"
    );
    return sums;
}

struct Lambdas
{
    NeoN::Vector<scalar> internal;
    NeoN::Vector<scalar> boundary;
};

// The nLimiterIter sweep of MULES::limiter / limiterCorr. psiMaxn/psiMinn are the bound
// right-hand sides (already multiplied by V/deltaT). Boundary lambda stays 1 (non-coupled
// patches) unless @p bOutletPhi is non-empty: then faces with bOutletPhi + bPhiCorr > SMALL^2
// are limited by their owner cell, as limiterCorr does for outlets.
Lambdas fctLambda(
    const NeoN::UnstructuredMesh& mesh,
    NeoN::View<const scalar> phiCorr,
    NeoN::View<const scalar> bPhiCorr,
    NeoN::View<const scalar> bOutletPhi,
    const LimiterSums& sums,
    int nLimiterIter
)
{
    const auto exec = mesh.exec();
    const auto nCells = mesh.nCells();
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    const auto psiMaxn = sums.psiMaxn.view();
    const auto psiMinn = sums.psiMinn.view();
    const auto sumPhip = sums.sumPhip.view();
    const auto mSumPhim = sums.mSumPhim.view();
    const bool limitOutlets = bOutletPhi.size() > 0;

    Lambdas lambdas {
        NeoN::Vector<scalar>(exec, mesh.nInternalFaces(), 1.0),
        NeoN::Vector<scalar>(exec, mesh.nBoundaryFaces(), 1.0)
    };
    auto lambda = lambdas.internal.view();
    auto bLambda = lambdas.boundary.view();
    NeoN::Vector<scalar> sumlPhipVec(exec, nCells, 0.0);
    NeoN::Vector<scalar> mSumlPhimVec(exec, nCells, 0.0);
    auto sumlPhip = sumlPhipVec.view();
    auto mSumlPhim = mSumlPhimVec.view();

    for (int j = 0; j < nLimiterIter; ++j)
    {
        // A local kernel, not NeoN::fill: an nvcc extended lambda instantiated from a NeoN
        // header template here can resolve to a foreign copy and call a null slot (#410).
        NeoN::parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx c) {
                sumlPhip[c] = 0.0;
                mSumlPhim[c] = 0.0;
            },
            "mules::zeroSuml"
        );
        NeoN::parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                const scalar lpc = lambda[f] * phiCorr[f];
                if (lpc > 0.0)
                {
                    Kokkos::atomic_add(&sumlPhip[own[f]], lpc);
                    Kokkos::atomic_add(&mSumlPhim[nei[f]], lpc);
                }
                else
                {
                    Kokkos::atomic_sub(&mSumlPhim[own[f]], lpc);
                    Kokkos::atomic_sub(&sumlPhip[nei[f]], lpc);
                }
            },
            "mules::sumlInternal"
        );
        NeoN::parallelFor(
            exec,
            {0, mesh.nBoundaryFaces()},
            NEON_LAMBDA(const localIdx bf) {
                const scalar lpc = bLambda[bf] * bPhiCorr[bf];
                if (lpc > 0.0)
                {
                    Kokkos::atomic_add(&sumlPhip[bOwn[bf]], lpc);
                }
                else
                {
                    Kokkos::atomic_sub(&mSumlPhim[bOwn[bf]], lpc);
                }
            },
            "mules::sumlBoundary"
        );
        NeoN::parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx c) {
                sumlPhip[c] = Kokkos::max(
                    Kokkos::min((sumlPhip[c] + psiMaxn[c]) / (mSumPhim[c] + rootVSmall), 1.0),
                    0.0
                );
                mSumlPhim[c] = Kokkos::max(
                    Kokkos::min((mSumlPhim[c] + psiMinn[c]) / (sumPhip[c] + rootVSmall), 1.0),
                    0.0
                );
            },
            "mules::lambdaCells"
        );
        // MULES reuses the sum arrays: lambdam = sumlPhip, lambdap = mSumlPhim.
        const auto lambdam = sumlPhip;
        const auto lambdap = mSumlPhim;
        NeoN::parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                if (phiCorr[f] > 0.0)
                {
                    lambda[f] =
                        Kokkos::min(lambda[f], Kokkos::min(lambdap[own[f]], lambdam[nei[f]]));
                }
                else
                {
                    lambda[f] =
                        Kokkos::min(lambda[f], Kokkos::min(lambdam[own[f]], lambdap[nei[f]]));
                }
            },
            "mules::lambdaInternal"
        );
        if (limitOutlets)
        {
            NeoN::parallelFor(
                exec,
                {0, mesh.nBoundaryFaces()},
                NEON_LAMBDA(const localIdx bf) {
                    if (bOutletPhi[bf] + bPhiCorr[bf] > small * small)
                    {
                        const scalar cellLambda =
                            (bPhiCorr[bf] > 0.0) ? lambdap[bOwn[bf]] : lambdam[bOwn[bf]];
                        bLambda[bf] = Kokkos::min(bLambda[bf], cellLambda);
                    }
                },
                "mules::lambdaOutlets"
            );
        }
    }
    return lambdas;
}

} // namespace

void mulesExplicitSolve(
    fvcc::VolumeField<scalar>& alpha,
    const fvcc::VolumeField<scalar>& alphaOld,
    const fvcc::SurfaceField<scalar>& phi,
    fvcc::SurfaceField<scalar>& alphaPhi,
    scalar deltaT,
    scalar psiMax,
    scalar psiMin,
    int nLimiterIter
)
{
    const auto& mesh = alpha.mesh();
    const auto exec = alpha.exec();
    const auto nCells = mesh.nCells();
    const auto nInt = mesh.nInternalFaces();
    const scalar rDeltaT = 1.0 / deltaT;

    correctBoundaryConditionsWithFlux(alpha, phi);

    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    const auto V = mesh.cellVolumes().view();
    // MULES uses psi (the current corrector's alpha) for phiBD and the local extrema, and psi0
    // (psi.oldTime()) for the bound right-hand sides and the update.
    auto a = alpha.internalVector().view();
    const auto a0 = alphaOld.internalVector().view();
    auto ap = alphaPhi.internalVector().view();
    const auto apB = alphaPhi.boundaryData().value().view();
    const auto phiI = phi.internalVector().view();

    // MULES::limit: phiBD = upwind flux; on non-coupled patches phiBD = alphaPhi, so the
    // boundary correction flux is zero.
    NeoN::Vector<scalar> phiBDVec(exec, nInt, 0.0);
    NeoN::Vector<scalar> phiCorrVec(exec, nInt, 0.0);
    NeoN::Vector<scalar> bPhiCorrVec(exec, mesh.nBoundaryFaces(), 0.0);
    auto phiBD = phiBDVec.view();
    auto phiCorr = phiCorrVec.view();
    NeoN::parallelFor(
        exec,
        {0, nInt},
        NEON_LAMBDA(const localIdx f) {
            const scalar w = (phiI[f] >= 0.0) ? 1.0 : 0.0;
            phiBD[f] = phiI[f] * faceValue(w, a[own[f]], a[nei[f]]);
            phiCorr[f] = ap[f] - phiBD[f];
        },
        "mules::phiBD"
    );

    LimiterSums sums = limiterSums(alpha, phiCorr, bPhiCorrVec.view(), psiMax, psiMin);

    NeoN::Vector<scalar> sumPhiBDVec(exec, nCells, 0.0);
    auto sumPhiBD = sumPhiBDVec.view();
    NeoN::parallelFor(
        exec,
        {0, nInt},
        NEON_LAMBDA(const localIdx f) {
            Kokkos::atomic_add(&sumPhiBD[own[f]], phiBD[f]);
            Kokkos::atomic_sub(&sumPhiBD[nei[f]], phiBD[f]);
        },
        "mules::sumPhiBDInternal"
    );
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { Kokkos::atomic_add(&sumPhiBD[bOwn[bf]], apB[bf]); },
        "mules::sumPhiBDBoundary"
    );

    auto psiMaxn = sums.psiMaxn.view();
    auto psiMinn = sums.psiMinn.view();
    NeoN::parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx c) {
            const scalar mx = Kokkos::min(psiMaxn[c], psiMax);
            const scalar mn = Kokkos::max(psiMinn[c], psiMin);
            psiMaxn[c] = V[c] * (rDeltaT * mx - rDeltaT * a0[c]) + sumPhiBD[c];
            psiMinn[c] = V[c] * (rDeltaT * a0[c] - rDeltaT * mn) - sumPhiBD[c];
        },
        "mules::boundRHS"
    );

    Lambdas lambdas = fctLambda(
        mesh,
        phiCorr,
        bPhiCorrVec.view(),
        NeoN::View<const scalar> {},
        sums,
        nLimiterIter
    );
    const auto lambda = lambdas.internal.view();

    NeoN::parallelFor(
        exec,
        {0, nInt},
        NEON_LAMBDA(const localIdx f) { ap[f] = phiBD[f] + lambda[f] * phiCorr[f]; },
        "mules::applyLimiter"
    );

    // MULES::explicitSolve: psi = (psi0*rDeltaT - surfaceIntegrate(alphaPhi))/rDeltaT.
    NeoN::Vector<scalar> divVec = integrateFlux(mesh, ap, apB);
    const auto div = divVec.view();
    NeoN::parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx c) { a[c] = (a0[c] * rDeltaT - div[c]) / rDeltaT; },
        "mules::update"
    );
    correctBoundaryConditionsWithFlux(alpha, phi);
}

void mulesCorrect(
    fvcc::VolumeField<scalar>& alpha,
    const fvcc::SurfaceField<scalar>& phi,
    fvcc::SurfaceField<scalar>& phiCorr,
    double deltaT,
    int nLimiterIter,
    double relaxOld
)
{
    const auto& mesh = alpha.mesh();
    const auto exec = alpha.exec();
    const auto nCells = mesh.nCells();
    const scalar rDeltaT = 1.0 / deltaT;
    const scalar psiMax = 1.0;
    const scalar psiMin = 0.0;

    NeoN::Vector<scalar> alphaPre(alpha.internalVector());

    const auto V = mesh.cellVolumes().view();
    auto a = alpha.internalVector().view();
    auto pc = phiCorr.internalVector().view();
    auto pcB = phiCorr.boundaryData().value().view();

    LimiterSums sums = limiterSums(alpha, pc, pcB, psiMax, psiMin);
    auto psiMaxn = sums.psiMaxn.view();
    auto psiMinn = sums.psiMinn.view();
    NeoN::parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx c) {
            const scalar mx = Kokkos::min(psiMaxn[c], psiMax);
            const scalar mn = Kokkos::max(psiMinn[c], psiMin);
            psiMaxn[c] = V[c] * (rDeltaT * mx - a[c] * rDeltaT);
            psiMinn[c] = V[c] * (-(rDeltaT * mn) + a[c] * rDeltaT);
        },
        "mulesCorr::boundRHS"
    );

    Lambdas lambdas =
        fctLambda(mesh, pc, pcB, phi.boundaryData().value().view(), sums, nLimiterIter);
    const auto lambda = lambdas.internal.view();
    const auto bLambda = lambdas.boundary.view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) { pc[f] *= lambda[f]; },
        "mulesCorr::lambdaInternal"
    );
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { pcB[bf] *= bLambda[bf]; },
        "mulesCorr::lambdaBoundary"
    );

    // MULES::correct: psi = (psi*rDeltaT - surfaceIntegrate(phiCorr))/rDeltaT.
    NeoN::Vector<scalar> divVec = integrateFlux(mesh, pc, pcB);
    const auto div = divVec.view();
    NeoN::parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx c) { a[c] = (a[c] * rDeltaT - div[c]) / rDeltaT; },
        "mulesCorr::apply"
    );

    if (relaxOld != 0.0)
    {
        const auto pre = alphaPre.view();
        const scalar w = relaxOld;
        NeoN::parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx c) { a[c] = (1.0 - w) * a[c] + w * pre[c]; },
            "mulesCorr::underRelax"
        );
    }
}

void correctBoundaryConditionsWithFlux(
    fvcc::VolumeField<scalar>& alpha,
    const fvcc::SurfaceField<scalar>& phi
)
{
    fvcc::BoundaryContext ctx;
    ctx.insert("phi", phi);
    alpha.correctBoundaryConditions(ctx);
}

} // namespace NeoFOAM
