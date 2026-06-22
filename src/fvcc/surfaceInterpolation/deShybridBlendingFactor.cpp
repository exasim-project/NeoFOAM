// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include <any>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/fvcc/surfaceInterpolation/deShybridBlendingFactor.hpp"

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

using NeoN::localIdx;
using NeoN::scalar;
using NeoN::Tensor;

namespace NeoFOAM
{

namespace
{

// Returns true and writes the value if the token holds any numeric type. OpenFOAM tokenises
// integer-looking values (e.g. "30", "0", "1") as labels and decimals (e.g. "0.65", "1.0e-3") as
// scalars, and the label/scalar widths vary by build, so all of double/float/int32/int64 are
// accepted.
bool tokenToScalar(const std::any& a, scalar& out)
{
    if (a.type() == typeid(double))
    {
        out = static_cast<scalar>(std::any_cast<double>(a));
        return true;
    }
    if (a.type() == typeid(float))
    {
        out = static_cast<scalar>(std::any_cast<float>(a));
        return true;
    }
    if (a.type() == typeid(std::int32_t))
    {
        out = static_cast<scalar>(std::any_cast<std::int32_t>(a));
        return true;
    }
    if (a.type() == typeid(std::int64_t))
    {
        out = static_cast<scalar>(std::any_cast<std::int64_t>(a));
        return true;
    }
    return false;
}

} // namespace

void computeDEShybridSigmaCell(
    const nnfvcc::VolumeField<Tensor>& gradU,
    const nnfvcc::VolumeField<scalar>& nut,
    const nnfvcc::VolumeField<scalar>& nu,
    const nnfvcc::VolumeField<scalar>& delta,
    const DEShybridCoefficients& coeffs,
    nnfvcc::VolumeField<scalar>& sigmaCell
)
{
    const auto exec = sigmaCell.exec();

    const auto [gradUV, nutV, nuV, deltaV] = NeoN::views(
        gradU.internalVector(), nut.internalVector(), nu.internalVector(), delta.internalVector()
    );
    auto sigmaV = sigmaCell.internalVector().view();

    // Hoist scalar constants out of the device lambda.
    const scalar CDES = coeffs.CDES;
    const scalar sigmaMin = coeffs.sigmaMin;
    const scalar sigmaMax = coeffs.sigmaMax;
    const scalar nutLim = coeffs.nutLim;
    const scalar CH1 = coeffs.CH1;
    const scalar CH2 = coeffs.CH2;
    const scalar CH3 = coeffs.CH3;
    const scalar Cs = coeffs.Cs;
    const scalar tau0 = coeffs.L0 / coeffs.U0;
    const scalar omegaLimTerm = coeffs.OmegaLim / tau0; // (OmegaLim/tau0)
    const scalar invTauTerm = scalar(0.1) / tau0;       // 0.1/tau0
    const scalar pow0p09 = Kokkos::pow(scalar(0.09), scalar(1.5));
    const scalar smallL0 = scalar(1.0e-15) * coeffs.L0; // OpenFOAM SMALL*L0

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(sigmaCell.internalVector().size())},
        NEON_LAMBDA(const localIdx i) {
            const Tensor& g = gradUV[i];

            // Strain-rate magnitude S = sqrt(2)*mag(symm(gradU)).
            const scalar s12 = scalar(0.5) * (g(0, 1) + g(1, 0));
            const scalar s13 = scalar(0.5) * (g(0, 2) + g(2, 0));
            const scalar s23 = scalar(0.5) * (g(1, 2) + g(2, 1));
            const scalar magSymmSq = g(0, 0) * g(0, 0) + g(1, 1) * g(1, 1) + g(2, 2) * g(2, 2)
                                   + scalar(2) * (s12 * s12 + s13 * s13 + s23 * s23);
            const scalar S = Kokkos::sqrt(scalar(2) * magSymmSq);

            // Vorticity magnitude Omega = sqrt(2)*mag(skew(gradU)) = 2*sqrt(sum a_ij^2).
            const scalar a12 = scalar(0.5) * (g(0, 1) - g(1, 0));
            const scalar a13 = scalar(0.5) * (g(0, 2) - g(2, 0));
            const scalar a23 = scalar(0.5) * (g(1, 2) - g(2, 1));
            const scalar Omega = scalar(2) * Kokkos::sqrt(a12 * a12 + a13 * a13 + a23 * a23);

            const scalar SsqPOsq = S * S + Omega * Omega;

            const scalar denomB =
                Kokkos::max(scalar(0.5) * SsqPOsq, omegaLimTerm * omegaLimTerm);
            const scalar B = CH3 * Omega * Kokkos::max(S, Omega) / denomB;
            const scalar B2 = B * B;
            const scalar gFun = Kokkos::tanh(B2 * B2);

            const scalar K = Kokkos::max(Kokkos::sqrt(scalar(0.5) * SsqPOsq), invTauTerm);

            const scalar cd = Cs * deltaV[i];
            const scalar nutEff =
                Kokkos::max(nutV[i], Kokkos::min(cd * cd * S, nutLim * nutV[i]));
            const scalar inner = (nutEff + nuV[i]) / (pow0p09 * K);
            const scalar lTurb = Kokkos::sqrt(Kokkos::max(inner, scalar(0)));

            const scalar A =
                CH2
                * Kokkos::max(
                    scalar(0), CDES * deltaV[i] / Kokkos::max(lTurb * gFun, smallL0) - scalar(0.5)
                );

            sigmaV[i] = Kokkos::max(sigmaMax * Kokkos::tanh(Kokkos::pow(A, CH1)), sigmaMin);
        },
        "DEShybrid::computeSigmaCell"
    );
}

void computeDEShybridBlendingFactor(
    const nnfvcc::VolumeField<Tensor>& gradU,
    const nnfvcc::VolumeField<scalar>& nut,
    const nnfvcc::VolumeField<scalar>& nu,
    const nnfvcc::VolumeField<scalar>& delta,
    const DEShybridCoefficients& coeffs,
    const nnfvcc::SurfaceInterpolation<scalar>& surfInterp,
    nnfvcc::SurfaceField<scalar>& sigmaFace
)
{
    const auto exec = gradU.exec();
    const auto& mesh = gradU.mesh();

    nnfvcc::VolumeField<scalar> sigmaCell(
        exec,
        "DEShybridSigmaCell",
        mesh,
        nnfvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
    );
    computeDEShybridSigmaCell(gradU, nut, nu, delta, coeffs, sigmaCell);
    sigmaCell.correctBoundaryConditions();
    surfInterp.interpolate(sigmaCell, sigmaFace);
}

DEShybridCoefficients readDEShybridCoefficients(NeoN::TokenList divSchemeTokens)
{
    const std::vector<std::any>& data = divSchemeTokens.tokens();

    // Collect the trailing run of numeric tokens (the coefficients), scanning from the end until a
    // non-numeric token (the LES-delta name, then the scheme names) is reached.
    std::vector<scalar> trailing;
    for (auto it = data.rbegin(); it != data.rend(); ++it)
    {
        scalar value;
        if (!tokenToScalar(*it, value))
        {
            break;
        }
        trailing.push_back(value);
    }

    if (trailing.size() < 6)
    {
        throw std::runtime_error(
            "readDEShybridCoefficients: expected at least 6 trailing numeric coefficients "
            "(CDES U0 L0 sigmaMin sigmaMax OmegaLim [nutLim]) in the DEShybrid div-scheme spec"
        );
    }

    // 'trailing' is reversed relative to the spec order; index from the back of the original spec.
    const auto n = trailing.size();
    DEShybridCoefficients c;
    c.CDES = trailing[n - 1];
    c.U0 = trailing[n - 2];
    c.L0 = trailing[n - 3];
    c.sigmaMin = trailing[n - 4];
    c.sigmaMax = trailing[n - 5];
    c.OmegaLim = trailing[n - 6];
    if (n >= 7)
    {
        c.nutLim = trailing[n - 7];
    }
    return c;
}

} // namespace NeoFOAM
