// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>

#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include "NeoFOAM/algorithms/vof/fluxes.hpp"
#include "detail.hpp"
#include "foamGeometry.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

namespace
{

using NeoN::localIdx;
using NeoN::scalar;

// alphaEqn.H's phic = cAlpha*|phi/magSf| on one face.
KOKKOS_INLINE_FUNCTION scalar compressionVelocity(scalar cAlpha, scalar phi, scalar magSf)
{
    return cAlpha * Kokkos::fabs(phi / magSf);
}

// fvc::flux(-fvc::flux(-phir, alpha2, linear), alpha1, linear) on one internal face,
// alpha2 = 1 - alpha1: (phir*alpha2_f)*alpha1_f. Negating twice is exact, so it is left out.
KOKKOS_INLINE_FUNCTION scalar
linearCompressionFaceFlux(scalar w, scalar phir, scalar alpha1P, scalar alpha1N)
{
    const scalar alpha2f = faceValue(w, 1.0 - alpha1P, 1.0 - alpha1N);
    const scalar alpha1f = faceValue(w, alpha1P, alpha1N);
    return (phir * alpha2f) * alpha1f;
}

// A calculated-BC surface field on @p alpha's mesh and executor.
fvcc::SurfaceField<scalar> newSurfaceField(const fvcc::VolumeField<scalar>& alpha, const char* name)
{
    return fvcc::SurfaceField<scalar>(
        alpha.exec(),
        name,
        alpha.mesh(),
        fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(alpha.mesh())
    );
}

} // namespace

// interfaceProperties::calculateK: nHatf = (gradAlphaf/(|gradAlphaf| + deltaN)) & Sf with
// gradAlphaf = linear interpolate(Gauss linear grad(alpha1)). The normalisation divides by
// deltaN ~ 1e-8/h, so the gradient must be summed in OpenFOAM's order to match it where
// gradAlpha is pure round-off.
void computeNHatf(
    RunTime& rt,
    const fvcc::VolumeField<scalar>& alpha1,
    fvcc::SurfaceField<scalar>& nHatf
)
{
    const auto exec = alpha1.exec();
    const auto& mesh = alpha1.mesh();
    const auto weights = foamGeometry(rt).weights.view();
    NeoN::Vector<NeoN::Vec3> gradVec = gaussLinearGrad(alpha1, weights);
    const auto g = gradVec.view();

    const auto nCells = mesh.nCells();
    const auto vView = mesh.cellVolumes().view();
    scalar sumV = 0.0;
    NeoN::parallelReduce(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx c, scalar& acc) { acc += vView[c]; },
        sumV
    );
    const scalar deltaN = 1e-8 / std::cbrt(sumV / static_cast<scalar>(nCells));

    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto sf = mesh.faceNormals().view();
    auto nh = nHatf.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) {
            const scalar w = weights[f];
            const NeoN::Vec3 gP = g[own[f]];
            const NeoN::Vec3 gN = g[nei[f]];
            const NeoN::Vec3 gf(
                faceValue(w, gP[0], gN[0]),
                faceValue(w, gP[1], gN[1]),
                faceValue(w, gP[2], gN[2])
            );
            nh[f] = (gf / (NeoN::mag(gf) + deltaN)) & sf[f];
        },
        "vof::nHatfInternal"
    );

    // Boundary: gaussGrad::correctBoundaryConditions on non-coupled patches,
    // g_b = g_P + n*(snGrad - n & g_P), snGrad = deltaCoeffs*(psi_b - psi_P).
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    const auto bSf = mesh.boundaryMesh().faceNormals().view();
    const auto bN = mesh.boundaryMesh().faceUnitNormals().view();
    const auto bDeltaCoeffs = mesh.boundaryMesh().deltaCoeffs().view();
    const auto aI = alpha1.internalVector().view();
    const auto aB = alpha1.boundaryData().value().view();
    auto bnh = nHatf.boundaryData().value().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) {
            const NeoN::Vec3 gP = g[bOwn[bf]];
            const NeoN::Vec3 n = bN[bf];
            const scalar snGrad = bDeltaCoeffs[bf] * (aB[bf] - aI[bOwn[bf]]);
            const NeoN::Vec3 gb = gP + n * (snGrad - (n & gP));
            bnh[bf] = (gb / (NeoN::mag(gb) + deltaN)) & bSf[bf];
        },
        "vof::nHatfBoundary"
    );
}

fvcc::SurfaceField<scalar> computeNHatf(RunTime& rt, const fvcc::VolumeField<scalar>& alpha1)
{
    fvcc::SurfaceField<scalar> nHatf = newSurfaceField(alpha1, "nHatf");
    computeNHatf(rt, alpha1, nHatf);
    return nHatf;
}

void vanLeerAlphaFlux(
    RunTime& rt,
    const fvcc::VolumeField<scalar>& alpha,
    const fvcc::SurfaceField<scalar>& phi,
    fvcc::SurfaceField<scalar>& flux
)
{
    const auto exec = alpha.exec();
    const auto& mesh = alpha.mesh();
    const FoamGeometry& geometry = foamGeometry(rt);
    const auto weights = geometry.weights.view();
    const auto delta = geometry.delta.view();
    NeoN::Vector<NeoN::Vec3> gradVec = gaussLinearGrad(alpha, weights);

    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto aI = alpha.internalVector().view();
    const auto gI = gradVec.view();
    const auto phiI = phi.internalVector().view();
    auto fI = flux.internalVector().view();

    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) {
            const localIdx o = own[f];
            const localIdx n = nei[f];
            const scalar phif = phiI[f];
            const NeoN::Vec3 d = delta[f];
            // NVDTVD::r; OpenFOAM's sign() is +1 for zero.
            const scalar gradf = aI[n] - aI[o];
            const scalar gradcf = (phif > 0.0) ? (d & gI[o]) : (d & gI[n]);
            scalar r;
            if (Kokkos::fabs(gradcf) >= 1000.0 * Kokkos::fabs(gradf))
            {
                const scalar signc = (gradcf >= 0.0) ? 1.0 : -1.0;
                const scalar signf = (gradf >= 0.0) ? 1.0 : -1.0;
                r = 2 * 1000 * signc * signf - 1;
            }
            else
            {
                r = 2 * (gradcf / gradf) - 1;
            }
            const scalar limiter = (r + Kokkos::fabs(r)) / (1 + Kokkos::fabs(r));
            const scalar upwindWeight = (phif >= 0.0) ? 1.0 : 0.0;
            const scalar w = limiter * weights[f] + (1.0 - limiter) * upwindWeight;
            fI[f] = phif * faceValue(w, aI[o], aI[n]);
        },
        "vof::vanLeerFluxInternal"
    );

    const auto aB = alpha.boundaryData().value().view();
    const auto phiB = phi.boundaryData().value().view();
    auto fB = flux.boundaryData().value().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { fB[bf] = phiB[bf] * aB[bf]; },
        "vof::vanLeerFluxBoundary"
    );
}

fvcc::SurfaceField<scalar> vanLeerAlphaFlux(
    RunTime& rt,
    const fvcc::VolumeField<scalar>& alpha,
    const fvcc::SurfaceField<scalar>& phi
)
{
    fvcc::SurfaceField<scalar> flux = newSurfaceField(alpha, "alphaPhiVanLeer");
    vanLeerAlphaFlux(rt, alpha, phi, flux);
    return flux;
}

void upwindAlphaFlux(
    const fvcc::VolumeField<scalar>& alpha,
    const fvcc::SurfaceField<scalar>& phi,
    fvcc::SurfaceField<scalar>& flux
)
{
    const auto exec = alpha.exec();
    const auto& mesh = alpha.mesh();
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto aI = alpha.internalVector().view();
    const auto phiI = phi.internalVector().view();
    auto fI = flux.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) {
            const scalar w = (phiI[f] >= 0.0) ? 1.0 : 0.0;
            fI[f] = phiI[f] * faceValue(w, aI[own[f]], aI[nei[f]]);
        },
        "vof::upwindFluxInternal"
    );
    const auto aB = alpha.boundaryData().value().view();
    const auto phiB = phi.boundaryData().value().view();
    auto fB = flux.boundaryData().value().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { fB[bf] = phiB[bf] * aB[bf]; },
        "vof::upwindFluxBoundary"
    );
}

fvcc::SurfaceField<scalar>
upwindAlphaFlux(const fvcc::VolumeField<scalar>& alpha, const fvcc::SurfaceField<scalar>& phi)
{
    fvcc::SurfaceField<scalar> flux = newSurfaceField(alpha, "alphaPhiUpwind");
    upwindAlphaFlux(alpha, phi, flux);
    return flux;
}

namespace
{

void linearCombination(
    scalar a,
    const NeoN::Vector<scalar>& x,
    scalar b,
    const NeoN::Vector<scalar>& y,
    NeoN::Vector<scalar>& out
)
{
    const auto xv = x.view();
    const auto yv = y.view();
    auto ov = out.view();
    NeoN::parallelFor(
        out.exec(),
        {0, out.size()},
        NEON_LAMBDA(const localIdx i) { ov[i] = a * xv[i] + b * yv[i]; },
        "vof::linearCombination"
    );
}

} // namespace

void linearCombination(
    scalar a,
    const fvcc::VolumeField<scalar>& x,
    scalar b,
    const fvcc::VolumeField<scalar>& y,
    fvcc::VolumeField<scalar>& out
)
{
    linearCombination(a, x.internalVector(), b, y.internalVector(), out.internalVector());
    linearCombination(
        a,
        x.boundaryData().value(),
        b,
        y.boundaryData().value(),
        out.boundaryData().value()
    );
}

void linearCombination(
    scalar a,
    const fvcc::SurfaceField<scalar>& x,
    scalar b,
    const fvcc::SurfaceField<scalar>& y,
    fvcc::SurfaceField<scalar>& out
)
{
    linearCombination(a, x.internalVector(), b, y.internalVector(), out.internalVector());
    linearCombination(
        a,
        x.boundaryData().value(),
        b,
        y.boundaryData().value(),
        out.boundaryData().value()
    );
}

void interfaceCompressionVelocity(
    const fvcc::SurfaceField<scalar>& phi,
    scalar cAlpha,
    fvcc::SurfaceField<scalar>& phic
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();
    const auto phiI = phi.internalVector().view();
    const auto magSf = mesh.faceAreas().view(); // |Sf|
    auto pI = phic.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) { pI[f] = compressionVelocity(cAlpha, phiI[f], magSf[f]); },
        "vof::compressionVelocity"
    );
    // NeoN has no coupled patches, so every boundary face is a non-coupled one.
    NeoN::fill(phic.boundaryData().value(), 0.0);
}

fvcc::SurfaceField<scalar>
interfaceCompressionVelocity(const fvcc::SurfaceField<scalar>& phi, scalar cAlpha)
{
    fvcc::SurfaceField<scalar> phic(
        phi.exec(),
        "phic",
        phi.mesh(),
        fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(phi.mesh())
    );
    interfaceCompressionVelocity(phi, cAlpha, phic);
    return phic;
}

void linearCompressionFlux(
    RunTime& rt,
    const fvcc::VolumeField<scalar>& alpha1,
    const fvcc::SurfaceField<scalar>& phir,
    fvcc::SurfaceField<scalar>& flux
)
{
    const auto exec = alpha1.exec();
    const auto& mesh = alpha1.mesh();
    const auto weights = foamGeometry(rt).weights.view();
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto aI = alpha1.internalVector().view();
    const auto phirI = phir.internalVector().view();
    auto fI = flux.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) {
            fI[f] = linearCompressionFaceFlux(weights[f], phirI[f], aI[own[f]], aI[nei[f]]);
        },
        "vof::linearCompressionFluxInternal"
    );
    const auto aB = alpha1.boundaryData().value().view();
    const auto phirB = phir.boundaryData().value().view();
    auto fB = flux.boundaryData().value().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { fB[bf] = (phirB[bf] * (1.0 - aB[bf])) * aB[bf]; },
        "vof::linearCompressionFluxBoundary"
    );
}

fvcc::SurfaceField<scalar> linearCompressionFlux(
    RunTime& rt,
    const fvcc::VolumeField<scalar>& alpha1,
    const fvcc::SurfaceField<scalar>& phir
)
{
    fvcc::SurfaceField<scalar> flux = newSurfaceField(alpha1, "alphaPhiCompression");
    linearCompressionFlux(rt, alpha1, phir, flux);
    return flux;
}

// alphaPhi += linearCompressionFlux(alpha1, phic*nHatf), face by face: the product and the sum
// are folded in so that a composition of the parts needs no phir or compression-flux field.
void addInterfaceCompressionFlux(
    RunTime& rt,
    const fvcc::VolumeField<scalar>& alpha1,
    const fvcc::SurfaceField<scalar>& phic,
    const fvcc::SurfaceField<scalar>& nHatf,
    fvcc::SurfaceField<scalar>& alphaPhi
)
{
    const auto exec = alpha1.exec();
    const auto& mesh = alpha1.mesh();
    const auto weights = foamGeometry(rt).weights.view();
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto aI = alpha1.internalVector().view();
    const auto phicI = phic.internalVector().view();
    const auto nhI = nHatf.internalVector().view();
    auto fI = alphaPhi.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) {
            const scalar phir = phicI[f] * nhI[f];
            fI[f] += linearCompressionFaceFlux(weights[f], phir, aI[own[f]], aI[nei[f]]);
        },
        "vof::addInterfaceCompressionFluxInternal"
    );
    const auto aB = alpha1.boundaryData().value().view();
    const auto phicB = phic.boundaryData().value().view();
    const auto nhB = nHatf.boundaryData().value().view();
    auto fB = alphaPhi.boundaryData().value().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) {
            const scalar phir = phicB[bf] * nhB[bf];
            fB[bf] += (phir * (1.0 - aB[bf])) * aB[bf];
        },
        "vof::addInterfaceCompressionFluxBoundary"
    );
}

// alphaEqn.H: alphaPhiUn = fvc::flux(phi, alpha1, "div(phi,alpha)")
//   + fvc::flux(-fvc::flux(-phir, alpha2, "div(phirb,alpha)"), alpha1, "div(phirb,alpha)").
// The face functions of interfaceCompressionVelocity and linearCompressionFlux fused into one
// kernel: composing the two would allocate phic, phir and the compression flux every call.
// phic is zeroed on non-coupled boundary faces, so the boundary flux is the vanLeer one.
fvcc::SurfaceField<scalar> alphaPhaseFlux(
    RunTime& rt,
    const fvcc::VolumeField<scalar>& alpha1,
    const fvcc::SurfaceField<scalar>& phi,
    double cAlpha
)
{
    fvcc::SurfaceField<scalar> flux = vanLeerAlphaFlux(rt, alpha1, phi);
    if (cAlpha == 0.0)
    {
        return flux;
    }

    const auto exec = alpha1.exec();
    const auto& mesh = alpha1.mesh();
    fvcc::SurfaceField<scalar> nHatf = computeNHatf(rt, alpha1);
    const auto weights = foamGeometry(rt).weights.view();
    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto aI = alpha1.internalVector().view();
    const auto phiI = phi.internalVector().view();
    const auto magSf = mesh.faceAreas().view(); // |Sf|
    const auto nh = nHatf.internalVector().view();
    auto fI = flux.internalVector().view();

    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) {
            const scalar phir = compressionVelocity(cAlpha, phiI[f], magSf[f]) * nh[f];
            fI[f] += linearCompressionFaceFlux(weights[f], phir, aI[own[f]], aI[nei[f]]);
        },
        "vof::alphaPhaseFluxCompression"
    );

    return flux;
}

} // namespace NeoFOAM
