// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/turbulenceModels/spalartAllmarasDDES.hpp"
#include "NeoFOAM/auxiliary/bound.hpp"
#include "NeoFOAM/auxiliary/writers.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"
#include "NeoFOAM/compatibility/fvSchemes.hpp"

#include "wallDist.H"
#include "nearWallDist.H"
#include "dimensionedScalar.H"

namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;

using NeoN::localIdx;
using NeoN::Tensor;
using NeoN::SymmTensor;

namespace NeoFOAM
{

// Named (not anonymous) so this model's SYCL device-kernel names stay unique across TUs — anonymous
// namespaces mangle to a shared `_GLOBAL__N_1` and would alias device images between models.
namespace saDdesDetail
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
            const scalar gSa = r + cw2 * (r6 - r);
            const scalar gSa6 = gSa * gSa * gSa * gSa * gSa * gSa;
            const scalar fw =
                gSa * Kokkos::pow((scalar(1) + cw36) / (gSa6 + cw36), scalar(1.0 / 6.0));

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

} // namespace saDdesDetail

using namespace saDdesDetail;

// ============================================================
// Private helpers: build NeoN fields from the OpenFOAM mesh
// ============================================================

namespace saDdesDetail
{

nnfvcc::VolumeField<scalar> buildWallDist(const NeoN::Executor& exec, MeshAdapter& mesh)
{
    Foam::wallDist y(mesh);
    return NeoFOAM::constructFrom(exec, mesh.nfMesh(), y.y());
}

nnfvcc::VolumeField<scalar> buildNearWallDist(const NeoN::Executor& exec, MeshAdapter& mesh)
{
    Foam::nearWallDist nwd(mesh);
    Foam::volScalarField ofField(
        Foam::IOobject(
            "nearWallDist",
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("zero", Foam::dimLength, Foam::scalar(0.0))
    );
    forAll(mesh.boundary(), patchi)
    {
        const Foam::scalarField& yf = nwd.y()[patchi];
        Foam::fvPatchScalarField& pf = ofField.boundaryFieldRef()[patchi];
        forAll(pf, facei)
        {
            pf[facei] = yf[facei];
        }
    }
    return NeoFOAM::constructFrom(exec, mesh.nfMesh(), ofField);
}

nnfvcc::VolumeField<scalar> buildDelta(const NeoN::Executor& exec, MeshAdapter& mesh)
{
    Foam::volScalarField ofField(
        Foam::IOobject(
            "delta",
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("delta", Foam::dimLength, Foam::scalar(0.0))
    );
    const Foam::scalarField& v = mesh.V();
    forAll(v, celli)
    {
        ofField.ref()[celli] = std::cbrt(v[celli]);
    }
    return NeoFOAM::constructFrom(exec, mesh.nfMesh(), ofField);
}

} // namespace saDdesDetail

// ============================================================
// Constructors
// ============================================================

SpalartAllmarasDDES::SpalartAllmarasDDES(RunTime& rt, const nnfvcc::VolumeField<scalar>& nu)
    : SpalartAllmarasDDES(rt.exec, rt.mesh, nu)
{
    // Replace the default Gauss grad operators with the gradSchemes-configured ones from
    // the RunTime cache, so grad(U)/grad(nuTilda) honour the chosen scheme (e.g. cellLimited)
    // and share a single instance with the solver's other grad(U) call sites.
    gradUOp_ = gradSchemePtr(rt, "grad(U)");
    gradNuTildaOp_ = gradSchemePtr(rt, "grad(nuTilda)");

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    if (solverDict.isDict("nuTilda"))
    {
        solverDict.subDict("nuTilda") = mapFvSolution(solverDict.subDict("nuTilda"));
    }
    if (solverDict.isDict("nuTildaFinal"))
    {
        solverDict.subDict("nuTildaFinal") = mapFvSolution(solverDict.subDict("nuTildaFinal"));
    }
}

SpalartAllmarasDDES::SpalartAllmarasDDES(
    const NeoN::Executor& exec,
    MeshAdapter& meshAdapter,
    const nnfvcc::VolumeField<scalar>& nu
)
    : SpalartAllmarasDDES(
        exec,
        meshAdapter.nfMesh(),
        nu,
        buildWallDist(exec, meshAdapter),
        buildNearWallDist(exec, meshAdapter),
        buildDelta(exec, meshAdapter)
    )
{
    // Read initial nuTilda and nut from the current time directory.
    const NeoN::UnstructuredMesh& nfMesh = mesh_;
    Foam::volScalarField ofNuTilda(
        Foam::IOobject(
            "nuTilda",
            meshAdapter.time().timeName(),
            meshAdapter,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        meshAdapter
    );
    Foam::volScalarField ofNut(
        Foam::IOobject(
            "nut",
            meshAdapter.time().timeName(),
            meshAdapter,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        meshAdapter
    );
    auto nuTildaInit = NeoFOAM::constructFrom(exec_, nfMesh, ofNuTilda);
    auto nutInit = NeoFOAM::constructFrom(exec_, nfMesh, ofNut);
    initialize(nuTildaInit, nutInit);
}

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
    , turbDb_()
    , nuTilda_(registerNuTilda(turbDb_, exec, mesh))
    , nut_(exec, "nut", mesh, fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh))
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
          // Proc-aware calculated BCs: gradU's processor tail must hold the NEIGHBOUR cell gradient
          // (filled by correctBoundaryConditions after gradTensor) so the proc-face viscous stress
          // in divDevReff interpolates the correct far-side gradient across the rank boundary.
          fvcc::createCalculatedProcBCs<nnfvcc::VolumeBoundary<Tensor>>(mesh)
      )
    , nuEff_(exec, "nuEff", mesh, fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh))
    , nuTildaEff_(
          exec,
          "DnuTildaEff",
          mesh,
          fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh)
      )
    , gradUOp_(nnfvcc::GradOperatorFactory<Vec3>::create(
          exec,
          mesh,
          NeoN::TokenList({std::string("Gauss"), std::string("linear")})
      ))
    , gradNuTildaOp_(nnfvcc::GradOperatorFactory<Vec3>::create(
          exec,
          mesh,
          NeoN::TokenList({std::string("Gauss"), std::string("linear")})
      ))
    , surfInterp_(exec, mesh, NeoN::TokenList({std::string("linear")}))
    , coeffs_()
    , cw1_(coeffs_.Cb1 / (coeffs_.kappa * coeffs_.kappa) + (1.0 + coeffs_.Cb2) / coeffs_.sigmaNut)
{
    surfInterp_.interpolate(nu_, surfNu_);
}

nnfvcc::VolumeField<scalar>& SpalartAllmarasDDES::registerNuTilda(
    NeoN::Database& db,
    const NeoN::Executor& exec,
    const NeoN::UnstructuredMesh& mesh
)
{
    auto& vectorCollection = fvcc::VectorCollection::instance(db, "VectorCollection");
    // Zero-initialised seed carrying the calculated BCs; registerVector copies it into the
    // collection and stamps the registration key. The registered copy (returned by reference) is
    // the one ddt/rotateOldTimes operate on; initialize() overwrites its data with the disk field.
    nnfvcc::VolumeField<scalar> seed(
        exec,
        "nuTilda",
        mesh,
        fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh)
    );
    return vectorCollection.registerVector<nnfvcc::VolumeField<scalar>>(
        fvcc::CreateFromExistingVector<nnfvcc::VolumeField<scalar>> {
            .name = "nuTilda",
            .field = seed,
            .timeIndex = 0,
            .iterationIndex = 0,
            .subCycleIndex = 0
        }
    );
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
    updateGradU(U);

    nnfvcc::SurfaceField<scalar> surfNut(
        exec_,
        "surfNut",
        mesh_,
        fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh_)
    );
    nnfvcc::SurfaceField<scalar> surfNuTilda(
        exec_,
        "surfNuTilda",
        mesh_,
        fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh_)
    );

    calcNuTildaDiffusionCoeff(nuTilda, surfNu_, surfNuTilda, nuTildaEff_);
    correctNut(nut, surfNut, nuEff_, nuTilda, nu_, surfNu_, U, nearWallDist_);
}

void SpalartAllmarasDDES::correct(
    const nnfvcc::VolumeField<Vec3>& U,
    nnfvcc::SurfaceField<scalar>& phi,
    nnfvcc::VolumeField<scalar>& nuTilda,
    nnfvcc::VolumeField<scalar>& nut,
    RunTime& rt
)
{
    updateGradU(U);

    nnfvcc::VolumeField<scalar> magSqrGradNuTilda(
        exec_,
        "magSqrGradNuTilda",
        mesh_,
        fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh_)
    );
    nnfvcc::VolumeField<scalar> production(
        exec_,
        "production",
        mesh_,
        fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh_)
    );
    nnfvcc::VolumeField<scalar> spCoeff(
        exec_,
        "spCoeff",
        mesh_,
        fvcc::createCalculatedBCs<nnfvcc::VolumeBoundary<scalar>>(mesh_)
    );
    nnfvcc::SurfaceField<scalar> surfNut(
        exec_,
        "surfNut",
        mesh_,
        fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh_)
    );
    nnfvcc::SurfaceField<scalar> surfNuTilda(
        exec_,
        "surfNuTilda",
        mesh_,
        fvcc::createCalculatedBCs<nnfvcc::SurfaceBoundary<scalar>>(mesh_)
    );
    // The Vector-out overload accumulates into the buffer without zeroing it first (it's meant to
    // be composed by callers who already own a zeroed accumulator), so use the value-returning
    // overload here instead, which zero-fills and includes the boundary gradient.
    nnfvcc::VolumeField<Vec3> gradNuTilda = gradNuTildaOp_->grad(nuTilda, NeoN::dsl::Coeff {});
    calcMagSqrVec(magSqrGradNuTilda, gradNuTilda);

    computeProdSpDDES(
        production,
        spCoeff,
        nuTilda,
        nu_,
        gradU_,
        wallDist_,
        delta_,
        magSqrGradNuTilda
    );

    PDE<scalar> nuTildaEqn(
        dsl::imp::ddt(nuTilda) + dsl::imp::div(phi, nuTilda)
            - dsl::imp::laplacian(nuTildaEff_, nuTilda) + dsl::imp::source(spCoeff, nuTilda)
            - dsl::exp::source(production),
        nuTilda,
        rt
    );
    nuTildaEqn.solve();

    // Bound nuTilda >= 0. bound() refills cells that undershot to zero or below from the
    // neighbourhood, as OpenFOAM does; a plain clamp pins them at the floor.
    bound(nuTilda, nuTildaMin_, boundCache_);
    nuTilda.correctBoundaryConditions();

    calcNuTildaDiffusionCoeff(nuTilda, surfNu_, surfNuTilda, nuTildaEff_);
    correctNut(nut, surfNut, nuEff_, nuTilda, nu_, surfNu_, U, nearWallDist_);
}

nnfvcc::SurfaceField<scalar>& SpalartAllmarasDDES::nuEff() { return nuEff_; }

const nnfvcc::VolumeField<scalar>& SpalartAllmarasDDES::nut() const { return nut_; }

const nnfvcc::VolumeField<Tensor>& SpalartAllmarasDDES::gradU() const { return gradU_; }

void SpalartAllmarasDDES::updateGradU(const nnfvcc::VolumeField<Vec3>& U)
{
    gradUOp_->gradTensor(U, gradU_, NeoN::dsl::Coeff {});
    // Exchange the neighbour-cell gradient into gradU's processor tail (proc patches carry the
    // processor BC; physical patches are 'calculated' no-ops, so their boundary gradient is kept).
    gradU_.correctBoundaryConditions();
}

void SpalartAllmarasDDES::initialize(
    const nnfvcc::VolumeField<scalar>& nuTildaInit,
    const nnfvcc::VolumeField<scalar>& nutInit
)
{
    nuTilda_.internalVector() = nuTildaInit.internalVector();
    nuTilda_.correctBoundaryConditions();
    nut_.internalVector() = nutInit.internalVector();
    nut_.correctBoundaryConditions();
}

void SpalartAllmarasDDES::validate(const nnfvcc::VolumeField<Vec3>& U)
{
    validate(U, nuTilda_, nut_);
}

void SpalartAllmarasDDES::correct(
    const nnfvcc::VolumeField<Vec3>& U,
    nnfvcc::SurfaceField<scalar>& phi,
    RunTime& rt
)
{
    correct(U, phi, nuTilda_, nut_, rt);
}

void SpalartAllmarasDDES::rotateOldTimes() { fvcc::rotateOldTimes(nuTilda_); }

void SpalartAllmarasDDES::write(MeshAdapter& mesh) const
{
    NeoFOAM::write(nuTilda_, mesh);
    NeoFOAM::write(nut_, mesh);
}

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
