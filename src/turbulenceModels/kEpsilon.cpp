// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/turbulenceModels/kEpsilon.hpp"
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
// Fused kernel: compute the production source Pk = nut * GbyNu0 and the
// dissipation-equation source/sp coefficients from the current k, ε, ν_t and
// ∇U. Mirrors OpenFOAM kEpsilon.C:237-292 — production-by-nu G/ν_t in the
// notation there.
// ---------------------------------------------------------------------------
void kernelComputeSources(
    const NeoN::Executor& exec,
    const NeoN::Vector<scalar>& kVec,
    const NeoN::Vector<scalar>& epsVec,
    const NeoN::Vector<scalar>& nutVec,
    const NeoN::Vector<Tensor>& gradUVec,
    NeoN::Vector<scalar>& PkVec,
    NeoN::Vector<scalar>& spKVec,
    NeoN::Vector<scalar>& epsSourceVec,
    NeoN::Vector<scalar>& spEpsVec,
    scalar Cmu,
    scalar C1,
    scalar C2
)
{
    const auto [kV, epsV, nutV, gradUV, PkV, spKV, epsSV, spEpsV] =
        NeoN::views(kVec, epsVec, nutVec, gradUVec, PkVec, spKVec, epsSourceVec, spEpsVec);

    const scalar rootVSmall = scalar(1e-30);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(kVec.size())},
        NEON_LAMBDA(const localIdx i) {
            const scalar k_i = Kokkos::max(kV[i], scalar(0));
            const scalar eps_i = Kokkos::max(epsV[i], rootVSmall);
            const scalar nut_i = nutV[i];

            // GbyNu0 = ∇U && devTwoSymm(∇U)
            //   = ∇U·∇U + ∇U·∇Uᵀ - (2/3)(div U)²
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

            // Production: G = ν_t · GbyNu0
            const scalar G_i = nut_i * GbyNu0_i;

            // k equation:
            //   source: +G
            //   implicit destruction (sp): ε/k     → spK = ε/k
            PkV[i] = G_i;
            spKV[i] = eps_i / Kokkos::max(k_i, rootVSmall);

            // ε equation:
            //   source: +C1 · Cμ · k · GbyNu0  (since C1·G·ε/k = C1·Cμ·k·GbyNu0
            //                                    using ν_t = Cμ·k²/ε)
            //   implicit destruction (sp): C2 · ε/k → spEps = C2 · ε/k
            epsSV[i] = C1 * Cmu * k_i * GbyNu0_i;
            spEpsV[i] = C2 * eps_i / Kokkos::max(k_i, rootVSmall);
        },
        "kEpsilon::computeSources"
    );
}

// ---------------------------------------------------------------------------
// Update interior ν_t = Cμ · k² / ε
// ---------------------------------------------------------------------------
void kernelCorrectNutInternal(
    const NeoN::Executor& exec,
    const NeoN::Vector<scalar>& kVec,
    const NeoN::Vector<scalar>& epsVec,
    NeoN::Vector<scalar>& nutVec,
    scalar Cmu
)
{
    const auto [kV, epsV, nutV_] = NeoN::views(kVec, epsVec, nutVec);
    const scalar rootVSmall = scalar(1e-30);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(kVec.size())},
        NEON_LAMBDA(const localIdx i) {
            const scalar k_i = Kokkos::max(kV[i], scalar(0));
            const scalar eps_i = Kokkos::max(epsV[i], rootVSmall);
            nutV_[i] = Cmu * k_i * k_i / eps_i;
        },
        "kEpsilon::correctNutInternal"
    );
}

// ---------------------------------------------------------------------------
// Face diffusivities: nuEff = ν + ν_t,  DkEff = ν + ν_t/σ_k, DepsEff = ν + ν_t/σ_ε
// ---------------------------------------------------------------------------
void kernelCalcDiffusivities(
    const NeoN::Executor& exec,
    const NeoN::Vector<scalar>& surfNuVec,
    const NeoN::Vector<scalar>& surfNutVec,
    NeoN::Vector<scalar>& nuEffVec,
    NeoN::Vector<scalar>& DkEffVec,
    NeoN::Vector<scalar>& DepsEffVec,
    scalar sigmaK,
    scalar sigmaEps,
    std::string label
)
{
    const auto [nuF, nutF, nuEffF, DkF, DepsF] =
        NeoN::views(surfNuVec, surfNutVec, nuEffVec, DkEffVec, DepsEffVec);

    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(nuEffVec.size())},
        NEON_LAMBDA(const localIdx f) {
            nuEffF[f] = nuF[f] + nutF[f];
            DkF[f] = nuF[f] + nutF[f] / sigmaK;
            DepsF[f] = nuF[f] + nutF[f] / sigmaEps;
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
        "kEpsilon::devRhoReff::boundary"
    );
}

} // namespace

// ============================================================
// Constructor
// ============================================================

KEpsilon::KEpsilon(
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
    , Pk_(exec, "Pk", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh))
    , spK_(exec, "spK", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh))
    , epsilonSource_(
          exec,
          "epsilonSource",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , spEpsilon_(
          exec,
          "spEpsilon",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
      )
    , surfNut_(
          exec,
          "surfNut",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , nuEff_(exec, "nuEff", mesh, fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh))
    , DkEffF_(exec, "DkEff", mesh, fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh))
    , DepsilonEffF_(
          exec,
          "DepsilonEff",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , gradOp_(exec, mesh)
    , surfInterp_(exec, mesh, NeoN::TokenList({std::string("linear")}))
    , coeffs_()
{
    surfInterp_.interpolate(nu_, surfNu_);

    // Populate nearWallDist_'s boundary face values by copying wallDist_'s
    // adjacent-cell internal value. Same pattern as KOmegaSST.
    {
        const auto wdInternal = wallDist_.internalVector().view();
        const auto faceOwners = mesh_.boundaryMesh().faceOwners().view();
        auto nwdBoundary = nearWallDist_.boundaryData().value().view();
        const auto nBoundaryFaces = static_cast<NeoN::localIdx>(nwdBoundary.size());
        NeoN::parallelFor(
            exec_,
            {0, nBoundaryFaces},
            NEON_LAMBDA(const NeoN::localIdx i) { nwdBoundary[i] = wdInternal[faceOwners[i]]; },
            "kEpsilon::initNearWallDistBoundary"
        );
    }
}

// ============================================================
// Public interface
// ============================================================

void KEpsilon::validate(
    const nnfvcc::VolumeField<Vec3>& U,
    const nnfvcc::VolumeField<scalar>& k,
    const nnfvcc::VolumeField<scalar>& epsilon,
    nnfvcc::VolumeField<scalar>& nut
)
{
    gradOp_.gradTensor(U, gradU_);
    gradU_.correctBoundaryConditions();

    computeSources(k, epsilon, nut, gradU_);
    correctNutInternal(k, epsilon, nut);

    // Wall-function BCs on nut (Spalding) need (U, nu, nearWallDist). The
    // epsilon wall function (when ported) will need (k, nu, nearWallDist).
    fvcc::BoundaryContext ctx;
    ctx.insert("U", U);
    ctx.insert("k", k);
    ctx.insert("nu", nu_);
    ctx.insert("nearWallDist", nearWallDist_);
    nut.correctBoundaryConditions(ctx);

    calcDiffusivities(nut);
}

void KEpsilon::correct(
    const nnfvcc::VolumeField<Vec3>& U,
    nnfvcc::SurfaceField<scalar>& phi,
    nnfvcc::VolumeField<scalar>& k,
    nnfvcc::VolumeField<scalar>& epsilon,
    nnfvcc::VolumeField<scalar>& nut,
    RunTime& rt
)
{
    // Compute ∇U once — reused for the production term
    gradOp_.gradTensor(U, gradU_);
    gradU_.correctBoundaryConditions();

    fvcc::BoundaryContext ctx;
    ctx.insert("U", U);
    ctx.insert("k", k);
    ctx.insert("nu", nu_);
    ctx.insert("nearWallDist", nearWallDist_);

    // Compute Pk, spK, εSource, spε using OLD ν_t
    computeSources(k, epsilon, nut, gradU_);

    // Refresh face diffusivities
    calcDiffusivities(nut);

    // OF feedback: overwrite Pk_ at wall-function cells with the epsilon wall
    // function's G formula. Mirrors OpenFOAM
    //   epsilonWallFunctionFvPatchScalarField::calculate() (epsilon.C:316-336):
    //     G[wall_cell] = (ν_t_w + ν_w) · |∂U/∂n|_w · C_µ^0.25 · √k / (κ · y)
    // The standard formula Pk = ν_t · GbyNu0 underestimates near-wall
    // production at refined meshes; the wall-function form uses the log-law
    // gradient directly. Without this overwrite the k equation lacks the
    // source that balances the wall-function ε boundary value.
    //
    // Identifies wall-function patches by name on epsilon's BC list.
    {
        const scalar Cmu25 = Kokkos::pow(coeffs_.Cmu, scalar(0.25));
        const scalar kappa_ = scalar(0.41); // matches epsilonWallFunction default
        const auto& epsilonBCs = epsilon.boundaryConditions();
        const auto faceOwnersV = mesh_.boundaryMesh().faceOwners().view();
        const auto deltaCoeffsV = mesh_.boundaryMesh().deltaCoeffs().view();
        const auto uInternalV = U.internalVector().view();
        const auto uBoundaryV = U.boundaryData().value().view();
        const auto nuBoundaryV = nu_.boundaryData().value().view();
        const auto nutBoundaryV = nut.boundaryData().value().view();
        const auto yBoundaryV = nearWallDist_.boundaryData().value().view();
        const auto kInternalV = k.internalVector().view();
        auto pkInternalV = Pk_.internalVector().view();

        for (NeoN::localIdx patchID = 0; patchID < static_cast<NeoN::localIdx>(epsilonBCs.size());
             ++patchID)
        {
            if (epsilonBCs[static_cast<size_t>(patchID)].name() != "epsilonWallFunction")
            {
                continue;
            }
            const auto [start, end] = epsilon.boundaryData().range(patchID);
            NeoN::parallelFor(
                exec_,
                {start, end},
                NEON_LAMBDA(const NeoN::localIdx i) {
                    const auto owner = faceOwnersV[i];
                    const NeoN::Vec3 uOwn = uInternalV[owner];
                    const NeoN::Vec3 uWall = uBoundaryV[i];
                    const scalar deltaInv = deltaCoeffsV[i];
                    const NeoN::Vec3 snGradU = (uWall - uOwn) * deltaInv;
                    const scalar magGradUw = NeoN::mag(snGradU);
                    const scalar nuw = nuBoundaryV[i];
                    const scalar nutw = nutBoundaryV[i];
                    const scalar y = yBoundaryV[i];
                    const scalar kc = Kokkos::max(kInternalV[owner], scalar(0));
                    const scalar gWall =
                        (nutw + nuw) * magGradUw * Cmu25 * Kokkos::sqrt(kc) / (kappa_ * y);
                    pkInternalV[owner] = gWall;
                },
                "kEpsilon::epsilonWFGFeedback"
            );
        }
    }

    // ----- epsilon equation -----
    // Solved BEFORE k so spK can be updated from the new ε to match OF's
    // sequencing (kEpsilon.C:251-291: ε first, then k).
    PDESolver<scalar> epsEqn(
        dsl::imp::ddt(epsilon) + dsl::imp::div(phi, epsilon)
            - dsl::imp::laplacian(DepsilonEffF_, epsilon) + dsl::imp::source(spEpsilon_, epsilon)
            - dsl::exp::source(epsilonSource_),
        epsilon,
        rt
    );
    epsEqn.solve();

    // Bound ε > 0
    {
        auto epsView = epsilon.internalVector().view();
        NeoN::parallelFor(
            exec_,
            {0, static_cast<localIdx>(epsilon.internalVector().size())},
            NEON_LAMBDA(const localIdx i) { epsView[i] = Kokkos::max(epsView[i], scalar(1e-10)); },
            "kEpsilon::boundEpsilon"
        );
        epsilon.correctBoundaryConditions(ctx);
    }

    // Update spK from the new ε (ε/k destruction term)
    {
        const scalar rootVSmall = scalar(1e-30);
        const auto kV = k.internalVector().view();
        const auto epsV = epsilon.internalVector().view();
        auto spKView = spK_.internalVector().view();
        NeoN::parallelFor(
            exec_,
            {0, static_cast<localIdx>(epsilon.internalVector().size())},
            NEON_LAMBDA(const localIdx i) {
                const scalar k_i = Kokkos::max(kV[i], rootVSmall);
                spKView[i] = epsV[i] / k_i;
            },
            "kEpsilon::updateSpKFromNewEpsilon"
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
            "kEpsilon::boundK"
        );
        k.correctBoundaryConditions(ctx);
    }

    // Update ν_t = Cμ k²/ε from new k, ε
    correctNutInternal(k, epsilon, nut);
    nut.correctBoundaryConditions(ctx);

    // Refresh surface diffusivities with new ν_t
    calcDiffusivities(nut);
}

nnfvcc::SurfaceField<scalar>& KEpsilon::nuEff() { return nuEff_; }

nnfvcc::SurfaceField<scalar>& KEpsilon::DkEff() { return DkEffF_; }

nnfvcc::SurfaceField<scalar>& KEpsilon::DepsilonEff() { return DepsilonEffF_; }

const nnfvcc::VolumeField<Tensor>& KEpsilon::gradU() const { return gradU_; }

NeoN::Vector<SymmTensor> KEpsilon::devRhoReff() const
{
    const localIdx nBF = static_cast<localIdx>(gradU_.boundaryData().value().size());
    NeoN::Vector<SymmTensor> result(exec_, nBF, NeoN::zero<SymmTensor>());
    kernelDevRhoReff(exec_, gradU_.boundaryData().value(), nuEff_.boundaryData().value(), result);
    return result;
}

// ============================================================
// Physics helpers
// ============================================================

void KEpsilon::computeSources(
    const nnfvcc::VolumeField<scalar>& k,
    const nnfvcc::VolumeField<scalar>& epsilon,
    const nnfvcc::VolumeField<scalar>& nut,
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU
)
{
    kernelComputeSources(
        exec_,
        k.internalVector(),
        epsilon.internalVector(),
        nut.internalVector(),
        gradU.internalVector(),
        Pk_.internalVector(),
        spK_.internalVector(),
        epsilonSource_.internalVector(),
        spEpsilon_.internalVector(),
        coeffs_.Cmu,
        coeffs_.C1,
        coeffs_.C2
    );
}

void KEpsilon::correctNutInternal(
    const nnfvcc::VolumeField<scalar>& k,
    const nnfvcc::VolumeField<scalar>& epsilon,
    nnfvcc::VolumeField<scalar>& nut
) const
{
    kernelCorrectNutInternal(
        exec_,
        k.internalVector(),
        epsilon.internalVector(),
        nut.internalVector(),
        coeffs_.Cmu
    );
}

void KEpsilon::calcDiffusivities(const nnfvcc::VolumeField<scalar>& nut)
{
    surfInterp_.interpolate(nut, surfNut_);

    kernelCalcDiffusivities(
        exec_,
        surfNu_.internalVector(),
        surfNut_.internalVector(),
        nuEff_.internalVector(),
        DkEffF_.internalVector(),
        DepsilonEffF_.internalVector(),
        coeffs_.sigmaK,
        coeffs_.sigmaEps,
        "kEpsilon::calcDiffusivities::internal"
    );
    kernelCalcDiffusivities(
        exec_,
        surfNu_.boundaryData().value(),
        surfNut_.boundaryData().value(),
        nuEff_.boundaryData().value(),
        DkEffF_.boundaryData().value(),
        DepsilonEffF_.boundaryData().value(),
        coeffs_.sigmaK,
        coeffs_.sigmaEps,
        "kEpsilon::calcDiffusivities::boundary"
    );
}

namespace
{

nnfvcc::VolumeField<scalar> buildWallDistKEps(const NeoN::Executor& exec, MeshAdapter& mesh)
{
    Foam::wallDist y(mesh);
    return NeoFOAM::constructFrom(exec, mesh.nfMesh(), y.y());
}

Foam::volScalarField readOFScalarFieldKEps(MeshAdapter& mesh, const std::string& fieldName)
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

KEpsilonModel::KEpsilonModel(RunTime& rt, const nnfvcc::VolumeField<scalar>& nu)
    : nu_(nu)
    , wallDist_(buildWallDistKEps(rt.exec, rt.mesh))
    , model_(rt.exec, rt.nfMesh, nu_, wallDist_)
{
    auto& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");

    k_ = &NeoFOAM::constructAndRegister(vc, rt, readOFScalarFieldKEps(rt.mesh, "k"), true);
    epsilon_ =
        &NeoFOAM::constructAndRegister(vc, rt, readOFScalarFieldKEps(rt.mesh, "epsilon"), true);
    nut_ = &NeoFOAM::constructAndRegister(vc, rt, readOFScalarFieldKEps(rt.mesh, "nut"), false);

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    for (const auto* f : {"k", "epsilon", "kFinal", "epsilonFinal"})
    {
        if (solverDict.isDict(f))
        {
            solverDict.subDict(f) = mapFvSolution(solverDict.subDict(f));
        }
    }
}

void KEpsilonModel::validate(const nnfvcc::VolumeField<Vec3>& U)
{
    model_.validate(U, *k_, *epsilon_, *nut_);
}

void KEpsilonModel::correct(
    const nnfvcc::VolumeField<Vec3>& U,
    nnfvcc::SurfaceField<scalar>& phi,
    RunTime& rt
)
{
    model_.correct(U, phi, *k_, *epsilon_, *nut_, rt);
}

nnfvcc::SurfaceField<scalar>& KEpsilonModel::nuEff() { return model_.nuEff(); }

const nnfvcc::VolumeField<scalar>& KEpsilonModel::nut() const { return *nut_; }

const nnfvcc::VolumeField<NeoN::Tensor>& KEpsilonModel::gradU() const { return model_.gradU(); }

void KEpsilonModel::rotateOldTimes()
{
    fvcc::rotateOldTimes(*k_);
    fvcc::rotateOldTimes(*epsilon_);
}

void KEpsilonModel::write(MeshAdapter& mesh) const
{
    NeoFOAM::write(*k_, mesh);
    NeoFOAM::write(*epsilon_, mesh);
    NeoFOAM::write(*nut_, mesh);
}

} // namespace NeoFOAM
