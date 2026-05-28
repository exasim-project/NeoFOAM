// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/turbulenceModels/spalartAllmarasDDES.hpp"

namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;

using NeoN::localIdx;
using NeoN::Tensor;
using NeoN::SymmTensor;

namespace NeoFOAM
{

namespace
{

void kernelCorrectNutInternal(
    const NeoN::Executor& exec,
    const NeoN::Vector<scalar>& nuTildeI,
    const NeoN::Vector<scalar>& nuI,
    NeoN::Vector<scalar>& nutI,
    scalar cv1Cubed
)
{
    const auto [nuTildeV, nuV, nutV] = NeoN::views(nuTildeI, nuI, nutI);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(nutI.size())},
        NEON_LAMBDA(const localIdx i) {
            const scalar chi = nuTildeV[i] / nuV[i];
            const scalar chi3 = chi * chi * chi;
            nutV[i] = nuTildeV[i] * chi3 / (chi3 + cv1Cubed);
        },
        "SA-DDES::correctNut::internal"
    );
}

void kernelAddViscosity(
    const NeoN::Executor& exec,
    const NeoN::Vector<scalar>& nuVec,
    const NeoN::Vector<scalar>& nutVec,
    NeoN::Vector<scalar>& nuEffVec,
    std::string label
)
{
    const auto [nuV, nutV, nuEffV] = NeoN::views(nuVec, nutVec, nuEffVec);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(nuEffVec.size())},
        NEON_LAMBDA(const localIdx f) { nuEffV[f] = nuV[f] + nutV[f]; },
        std::move(label)
    );
}

void kernelNuTildaDiffCoeff(
    const NeoN::Executor& exec,
    const NeoN::Vector<scalar>& nuVec,
    const NeoN::Vector<scalar>& nuTildeVec,
    NeoN::Vector<scalar>& nuTildeEffVec,
    scalar invSigmaNut,
    std::string label
)
{
    const auto [nuV, nuTildeV, nuTildeEffV] = NeoN::views(nuVec, nuTildeVec, nuTildeEffVec);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(nuTildeEffVec.size())},
        NEON_LAMBDA(const localIdx f) { nuTildeEffV[f] = invSigmaNut * (nuV[f] + nuTildeV[f]); },
        std::move(label)
    );
}

void kernelMagSqrVec(
    const NeoN::Executor& exec,
    const NeoN::Vector<Vec3>& inVec,
    NeoN::Vector<scalar>& magVec
)
{
    const auto [valueV, magV] = NeoN::views(inVec, magVec);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(magVec.size())},
        NEON_LAMBDA(const localIdx i) {
            magV[i] = valueV[i][0] * valueV[i][0] + valueV[i][1] * valueV[i][1]
                    + valueV[i][2] * valueV[i][2];
        },
        "SA-DDES::magSqrGradNuTilde::internal"
    );
}

void kernelComputeProdSp(
    const NeoN::Executor& exec,
    const NeoN::Vector<scalar>& nuTildeVec,
    const NeoN::Vector<scalar>& nuVec,
    const NeoN::Vector<Tensor>& gradUVec,
    const NeoN::Vector<scalar>& wallDistVec,
    const NeoN::Vector<scalar>& deltaVec,
    const NeoN::Vector<scalar>& gradNuTildeMagSqrVec,
    NeoN::Vector<scalar>& productionVec,
    NeoN::Vector<scalar>& spCoeffVec,
    scalar cv13,
    scalar kappa2,
    scalar cb1,
    scalar cb2S,
    scalar cw1,
    scalar cw2,
    scalar cw36,
    scalar fdCoef,
    scalar cdes,
    scalar cs,
    scalar fwStar
)
{
    const auto
        [nuTildeV, nuV, gradUV, wallDistV, deltaV, gradNuTildeMagSqrV, productionV, spCoeffV] =
            NeoN::views(
                nuTildeVec,
                nuVec,
                gradUVec,
                wallDistVec,
                deltaVec,
                gradNuTildeMagSqrVec,
                productionVec,
                spCoeffVec
            );

    const scalar rootvsmall = scalar(1e-30);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(nuTildeVec.size())},
        NEON_LAMBDA(const localIdx i) {
            const scalar nuTilda = nuTildeV[i];
            const scalar nu = nuV[i];

            const scalar chi = nuTilda / nu;
            const scalar chi2 = chi * chi;
            const scalar chi3 = chi2 * chi;
            const scalar fv1 = chi3 / (chi3 + cv13);
            const scalar fv2 = scalar(1) - chi / (scalar(1) + chi * fv1);

            const Tensor& g = gradUV[i];

            scalar magGradUSq = scalar(0);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    magGradUSq += g(r, c) * g(r, c);
            const scalar magGradU = Kokkos::sqrt(magGradUSq);

            // Vorticity from antisymmetric part: omega_ij = 0.5*(g_ij - g_ji)
            const scalar a12 = scalar(0.5) * (g(0, 1) - g(1, 0));
            const scalar a13 = scalar(0.5) * (g(0, 2) - g(2, 0));
            const scalar a23 = scalar(0.5) * (g(1, 2) - g(2, 1));
            const scalar omega = scalar(2.0) * Kokkos::sqrt(a12 * a12 + a13 * a13 + a23 * a23);

            const scalar dWall = Kokkos::max(wallDistV[i], rootvsmall);
            const scalar rD = Kokkos::min(
                (nu + nuTilda * fv1) / (kappa2 * dWall * dWall * Kokkos::max(magGradU, rootvsmall)),
                scalar(10)
            );
            const scalar fD = scalar(1) - Kokkos::tanh(Kokkos::pow(fdCoef * rD, 3));

            const scalar psi = Kokkos::sqrt(Kokkos::min(
                scalar(100),
                (scalar(1) - cb1 / (cw1 * kappa2 * fwStar) * fv2) / Kokkos::max(fv1, rootvsmall)
            ));

            const scalar dTilde = Kokkos::max(
                dWall - fD * Kokkos::max(dWall - psi * cdes * deltaV[i], scalar(0)),
                rootvsmall
            );
            const scalar invSqrdTilde = scalar(1) / (dTilde * dTilde);

            const scalar sTilde =
                Kokkos::max(omega + fv2 * nuTilda * invSqrdTilde / kappa2, cs * omega);

            const scalar r = Kokkos::min(
                nuTilda / (Kokkos::max(sTilde, rootvsmall) * kappa2 * dTilde * dTilde),
                scalar(10)
            );
            const scalar r6 = r * r * r * r * r * r;
            const scalar g = r + cw2 * (r6 - r);
            const scalar g6 = g * g * g * g * g * g;
            const scalar fw = g * Kokkos::pow((scalar(1) + cw36) / (g6 + cw36), scalar(1.0 / 6.0));

            productionV[i] = cb1 * sTilde * nuTilda + cb2S * gradNuTildeMagSqrV[i];
            spCoeffV[i] = cw1 * fw * nuTilda * invSqrdTilde;
        },
        "SA-DDES::prod+Sp+omega+magGradU/fused/Tensor"
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
        "SA-DDES::devRhoReff::boundary"
    );
}

} // namespace

// ============================================================
// Constructor
// ============================================================

SpalartAllmarasDDES::SpalartAllmarasDDES(
    const NeoN::Executor& exec,
    const NeoN::UnstructuredMesh& mesh,
    const nnfvcc::VolumeField<scalar>& nu,
    const nnfvcc::VolumeField<scalar>& wallDist,
    const nnfvcc::VolumeField<scalar>& nearWallDist,
    const nnfvcc::VolumeField<scalar>& delta
)
    : exec_(exec)
    , mesh_(mesh)
    , nu_(nu)
    , wallDist_(wallDist)
    , nearWallDist_(nearWallDist)
    , delta_(delta)
    , surfNu_(
          exec,
          "surfNu",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , gradU_(exec, "gradU", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<Tensor>>(mesh))
    , gradNuTilda_(
          exec,
          "gradNuTilda",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<Vec3>>(mesh)
      )
    , magSqrGradNuTilda_(
          exec,
          "magSqrGradNuTilda",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , production_(
          exec,
          "production",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , spCoeff_(
          exec,
          "spCoeff",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , surfNut_(
          exec,
          "surfNut",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , surfNuTilda_(
          exec,
          "surfNuTilda",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , nuEff_(exec, "nuEff", mesh, fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh))
    , nuTildaEff_(
          exec,
          "DnuTildaEff",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , gradOp_(exec, mesh)
    , surfInterp_(exec, mesh, NeoN::TokenList({std::string("linear")}))
    , coeffs_()
    , cw1_(coeffs_.Cb1 / (coeffs_.kappa * coeffs_.kappa) + (1.0 + coeffs_.Cb2) / coeffs_.sigmaNut)
{
    surfInterp_.interpolate(nu_, surfNu_);
}

// ============================================================
// Public interface
// ============================================================

void SpalartAllmarasDDES::validate(
    const nnfvcc::VolumeField<Vec3>& U,
    nnfvcc::VolumeField<scalar>& nuTilda,
    nnfvcc::VolumeField<scalar>& nut
)
{
    gradOp_.gradTensor(U, gradU_);
    calcNuTildaDiffusionCoeff(nuTilda, surfNu_, surfNuTilda_, nuTildaEff_);
    correctNut(nut, surfNut_, nuEff_, nuTilda, nu_, surfNu_, U, nearWallDist_);
}

void SpalartAllmarasDDES::correct(
    const nnfvcc::VolumeField<Vec3>& U,
    nnfvcc::SurfaceField<scalar>& phi,
    nnfvcc::VolumeField<scalar>& nuTilda,
    nnfvcc::VolumeField<scalar>& nut,
    RunTime& rt
)
{
    gradOp_.gradTensor(U, gradU_);
    gradOp_.grad(nuTilda, gradNuTilda_);
    calcMagSqrVec(magSqrGradNuTilda_, gradNuTilda_);

    computeProdSpDDES(
        production_,
        spCoeff_,
        nuTilda,
        nu_,
        gradU_,
        wallDist_,
        delta_,
        magSqrGradNuTilda_
    );

    PDESolver<scalar> nuTildaEqn(
        dsl::imp::ddt(nuTilda) + dsl::imp::div(phi, nuTilda)
            - dsl::imp::laplacian(nuTildaEff_, nuTilda) + dsl::imp::source(spCoeff_, nuTilda)
            - dsl::exp::source(production_),
        nuTilda,
        rt
    );
    nuTildaEqn.solve();

    calcNuTildaDiffusionCoeff(nuTilda, surfNu_, surfNuTilda_, nuTildaEff_);
    correctNut(nut, surfNut_, nuEff_, nuTilda, nu_, surfNu_, U, nearWallDist_);
}

nnfvcc::SurfaceField<scalar>& SpalartAllmarasDDES::nuEff() { return nuEff_; }

nnfvcc::SurfaceField<scalar>& SpalartAllmarasDDES::nuTildaEff() { return nuTildaEff_; }

const nnfvcc::VolumeField<Tensor>& SpalartAllmarasDDES::gradU() const { return gradU_; }

NeoN::Vector<SymmTensor> SpalartAllmarasDDES::devRhoReff() const
{
    const localIdx nBF = static_cast<localIdx>(gradU_.boundaryData().value().size());
    NeoN::Vector<SymmTensor> result(exec_, nBF, NeoN::zero<SymmTensor>());
    kernelDevRhoReff(exec_, gradU_.boundaryData().value(), nuEff_.boundaryData().value(), result);
    return result;
}

// ============================================================
// Private physics helpers (thin wrappers, delegate to kernels)
// ============================================================

void SpalartAllmarasDDES::correctNut(
    nnfvcc::VolumeField<scalar>& nutField,
    nnfvcc::SurfaceField<scalar>& nutF,
    nnfvcc::SurfaceField<scalar>& nuEffF,
    const nnfvcc::VolumeField<scalar>& nuTilde,
    const nnfvcc::VolumeField<scalar>& nu,
    const nnfvcc::SurfaceField<scalar>& nuF,
    const nnfvcc::VolumeField<Vec3>& u,
    const nnfvcc::VolumeField<scalar>& nearWallDist
) const
{
    const scalar cv1Cubed = coeffs_.Cv1 * coeffs_.Cv1 * coeffs_.Cv1;
    kernelCorrectNutInternal(
        exec_,
        nuTilde.internalVector(),
        nu.internalVector(),
        nutField.internalVector(),
        cv1Cubed
    );

    fvcc::BoundaryContext ctx;
    ctx.insert("U", u);
    ctx.insert("nu", nu);
    ctx.insert("nearWallDist", nearWallDist);
    nutField.correctBoundaryConditions(ctx);
    surfInterp_.interpolate(nutField, nutF);

    kernelAddViscosity(
        exec_,
        nuF.internalVector(),
        nutF.internalVector(),
        nuEffF.internalVector(),
        "SA-DDES::correctNut::nuEffF"
    );
    kernelAddViscosity(
        exec_,
        nuF.boundaryData().value(),
        nutF.boundaryData().value(),
        nuEffF.boundaryData().value(),
        "SA-DDES::nuEffF::boundary"
    );

    nuEffF.name = "nuEff";
}

void SpalartAllmarasDDES::calcNuTildaDiffusionCoeff(
    nnfvcc::VolumeField<scalar>& nuTilde,
    const nnfvcc::SurfaceField<scalar>& nuF,
    nnfvcc::SurfaceField<scalar>& surfNuTilde,
    nnfvcc::SurfaceField<scalar>& nuTildeEffF
) const
{
    nuTilde.correctBoundaryConditions();
    surfInterp_.interpolate(nuTilde, surfNuTilde);

    const scalar invSigmaNut = scalar(1) / coeffs_.sigmaNut;

    kernelNuTildaDiffCoeff(
        exec_,
        nuF.internalVector(),
        surfNuTilde.internalVector(),
        nuTildeEffF.internalVector(),
        invSigmaNut,
        "SA-DDES::calcNuTildaDiffusionCoeff::internal"
    );
    kernelNuTildaDiffCoeff(
        exec_,
        nuF.boundaryData().value(),
        surfNuTilde.boundaryData().value(),
        nuTildeEffF.boundaryData().value(),
        invSigmaNut,
        "SA-DDES::calcNuTildaDiffusionCoeff::boundary"
    );

    nuTildeEffF.name = "DnuTildaEff";
}

void SpalartAllmarasDDES::calcMagSqrVec(
    nnfvcc::VolumeField<scalar>& magSqr,
    const nnfvcc::VolumeField<Vec3>& in
) const
{
    kernelMagSqrVec(exec_, in.internalVector(), magSqr.internalVector());
}

void SpalartAllmarasDDES::computeProdSpDDES(
    nnfvcc::VolumeField<scalar>& productionField,
    nnfvcc::VolumeField<scalar>& spCoeffField,
    const nnfvcc::VolumeField<scalar>& nuTildeField,
    const nnfvcc::VolumeField<scalar>& nuField,
    const nnfvcc::VolumeField<Tensor>& gradUField,
    const nnfvcc::VolumeField<scalar>& wallDistanceField,
    const nnfvcc::VolumeField<scalar>& deltaField,
    const nnfvcc::VolumeField<scalar>& gradNuTildeMagSqrField
) const
{
    kernelComputeProdSp(
        exec_,
        nuTildeField.internalVector(),
        nuField.internalVector(),
        gradUField.internalVector(),
        wallDistanceField.internalVector(),
        deltaField.internalVector(),
        gradNuTildeMagSqrField.internalVector(),
        productionField.internalVector(),
        spCoeffField.internalVector(),
        coeffs_.Cv1 * coeffs_.Cv1 * coeffs_.Cv1,
        coeffs_.kappa * coeffs_.kappa,
        coeffs_.Cb1,
        coeffs_.Cb2 / coeffs_.sigmaNut,
        cw1_,
        coeffs_.Cw2,
        std::pow(coeffs_.Cw3, 6),
        coeffs_.fdCoef,
        coeffs_.Cdes,
        coeffs_.Cs,
        coeffs_.fwStar
    );
}

} // namespace NeoFOAM
