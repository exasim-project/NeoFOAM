// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // main is provided by neofoam_catch_main

#include <cmath>

#include "common.hpp"

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "wallDist.H"
#include "LESModel.H"
#include "surfaceInterpolationScheme.H"
#include "blendedSchemeBase.H"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using Scalar = NeoN::scalar;
using Vec3 = NeoN::Vec3;
using Tensor = NeoN::Tensor;
using VolScalar = fvcc::VolumeField<Scalar>;
using VolTensor = fvcc::VolumeField<Tensor>;
using SurfScalar = fvcc::SurfaceField<Scalar>;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

namespace
{

// Host replication of the Travin et al. blending factor for the simple-shear case S == Omega == g.
Scalar analyticSigma(Scalar S, Scalar Omega, Scalar nut, Scalar nu, Scalar delta, const nf::DEShybridCoefficients& c)
{
    const Scalar tau0 = c.L0 / c.U0;
    const Scalar SsqPOsq = S * S + Omega * Omega;
    const Scalar omegaLimTerm = c.OmegaLim / tau0;
    const Scalar denomB = std::max(Scalar(0.5) * SsqPOsq, omegaLimTerm * omegaLimTerm);
    const Scalar B = c.CH3 * Omega * std::max(S, Omega) / denomB;
    const Scalar gFun = std::tanh(B * B * B * B);
    const Scalar K = std::max(std::sqrt(Scalar(0.5) * SsqPOsq), Scalar(0.1) / tau0);
    const Scalar cd = c.Cs * delta;
    const Scalar nutEff = std::max(nut, std::min(cd * cd * S, c.nutLim * nut));
    const Scalar pow0p09 = std::pow(Scalar(0.09), Scalar(1.5));
    const Scalar lTurb = std::sqrt(std::max((nutEff + nu) / (pow0p09 * K), Scalar(0)));
    const Scalar smallL0 = Scalar(1.0e-15) * c.L0;
    const Scalar A =
        c.CH2 * std::max(Scalar(0), c.CDES * delta / std::max(lTurb * gFun, smallL0) - Scalar(0.5));
    return std::max(c.sigmaMax * std::tanh(std::pow(A, c.CH1)), c.sigmaMin);
}

template<typename FieldT, typename ValueT>
FieldT makeFilled(const NeoN::Executor& exec, const NeoN::UnstructuredMesh& mesh, const std::string& nm, ValueT v)
{
    using Boundary = std::conditional_t<
        std::is_same_v<FieldT, SurfScalar>, fvcc::SurfaceBoundary<typename FieldT::VectorValueType>,
        fvcc::VolumeBoundary<typename FieldT::VectorValueType>>;
    FieldT f(exec, nm, mesh, fvcc::createCalculatedBCs<Boundary>(mesh));
    fill(f.internalVector(), v);
    fill(f.boundaryData().value(), v);
    return f;
}

} // namespace

// Validates the DEShybrid blending-factor kernel against the analytic Travin formula for a uniform
// simple-shear gradient (S == Omega), plus the physical limits in the LES filter width delta.
TEST_CASE("DEShybrid sigma kernel matches the analytic Travin formula")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto meshAdapter = NeoFOAM::createMesh(exec, runTime);
    const auto& nfMesh = meshAdapter->nfMesh();

    const Scalar gamma = 2.0; // simple shear du/dy -> S == Omega == gamma
    const Scalar nutVal = 1.0e-3;
    const Scalar nuVal = 1.0e-5;
    const Scalar deltaVal = 0.3; // chosen so sigma lands strictly inside (sigmaMin, sigmaMax)

    nf::DEShybridCoefficients coeffs;
    coeffs.CDES = 0.65;
    coeffs.U0 = 10.0;
    coeffs.L0 = 1.0;
    coeffs.sigmaMin = 0.05;
    coeffs.sigmaMax = 0.9;
    coeffs.OmegaLim = 1.0e-3;
    coeffs.nutLim = 1.0;

    auto gradU = makeFilled<VolTensor>(exec, nfMesh, "gradU", Tensor(0, gamma, 0, 0, 0, 0, 0, 0, 0));
    auto nut = makeFilled<VolScalar>(exec, nfMesh, "nut", nutVal);
    auto nu = makeFilled<VolScalar>(exec, nfMesh, "nu", nuVal);
    auto delta = makeFilled<VolScalar>(exec, nfMesh, "delta", deltaVal);
    auto sigmaCell = makeFilled<VolScalar>(exec, nfMesh, "sigmaCell", Scalar(0));

    nf::computeDEShybridSigmaCell(gradU, nut, nu, delta, coeffs, sigmaCell);

    const Scalar expected = analyticSigma(gamma, gamma, nutVal, nuVal, deltaVal, coeffs);
    INFO("expected sigma: " << expected);
    REQUIRE(expected > coeffs.sigmaMin);
    REQUIRE(expected < coeffs.sigmaMax);

    auto sH = sigmaCell.internalVector().copyToHost();
    for (NeoN::localIdx i = 0; i < sH.size(); ++i)
    {
        REQUIRE(sH.view()[i] == Catch::Approx(expected).margin(1e-12));
    }

    // delta -> 0 : the LES length scale collapses, A -> 0, sigma -> sigmaMin (pure scheme 1).
    auto deltaZero = makeFilled<VolScalar>(exec, nfMesh, "deltaZero", Scalar(0));
    nf::computeDEShybridSigmaCell(gradU, nut, nu, deltaZero, coeffs, sigmaCell);
    auto sZeroH = sigmaCell.internalVector().copyToHost();
    for (NeoN::localIdx i = 0; i < sZeroH.size(); ++i)
    {
        REQUIRE(sZeroH.view()[i] == Catch::Approx(coeffs.sigmaMin).margin(1e-12));
    }

    // very large delta : A -> large, tanh saturates, sigma -> sigmaMax (pure scheme 2).
    auto deltaBig = makeFilled<VolScalar>(exec, nfMesh, "deltaBig", Scalar(1.0e6));
    nf::computeDEShybridSigmaCell(gradU, nut, nu, deltaBig, coeffs, sigmaCell);
    auto sBigH = sigmaCell.internalVector().copyToHost();
    for (NeoN::localIdx i = 0; i < sBigH.size(); ++i)
    {
        REQUIRE(sBigH.view()[i] == Catch::Approx(coeffs.sigmaMax).margin(1e-9));
    }
}

// Validates the solver-facing injection path: a per-face sigma registered in the source field's
// database under "<src>DEShybridBlendingFactor" is picked up by NeoN's DEShybrid scheme, so the
// interpolated value equals the explicit blend (1-sigma)*linear + sigma*linearUpwind.
TEST_CASE("DEShybrid scheme consumes a database-registered blending factor")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto meshAdapter = NeoFOAM::createMesh(exec, runTime);
    const auto& nfMesh = meshAdapter->nfMesh();

    NeoN::Database db;
    auto& collection = fvcc::VectorCollection::instance(db, "fieldCollection");

    // Register the source field "src" in the collection.
    VolScalar srcSeed = makeFilled<VolScalar>(exec, nfMesh, "src", Scalar(0));
    {
        auto h = srcSeed.internalVector().copyToHost();
        for (NeoN::localIdx i = 0; i < h.size(); ++i) h.view()[i] = 1.0 + 0.5 * static_cast<Scalar>(i);
        srcSeed.internalVector() = h.copyToExecutor(exec);
    }
    auto& src = collection.registerVector<VolScalar>(
        fvcc::CreateFromExistingVector<VolScalar> {.name = "src", .field = srcSeed}
    );
    src.correctBoundaryConditions();

    // Register the sigma field under the name DEShybrid looks up.
    const Scalar sigmaVal = 0.4;
    SurfScalar sigmaSeed = makeFilled<SurfScalar>(exec, nfMesh, "srcDEShybridBlendingFactor", sigmaVal);
    collection.registerVector<SurfScalar>(
        fvcc::CreateFromExistingVector<SurfScalar> {
            .name = "srcDEShybridBlendingFactor", .field = sigmaSeed
        }
    );

    auto flux = makeFilled<SurfScalar>(exec, nfMesh, "flux", Scalar(1));

    // Reference sub-schemes.
    auto lin =
        fvcc::SurfaceInterpolation<Scalar>(exec, nfMesh, NeoN::Input(NeoN::TokenList({std::string("linear")})));
    auto lup = fvcc::SurfaceInterpolation<Scalar>(
        exec, nfMesh, NeoN::Input(NeoN::TokenList({std::string("linearUpwind"), std::string("Gauss")}))
    );

    // DEShybrid kernel with member sigma left at its default (0): if the DB lookup failed it would
    // reduce to linear, so matching the sigma=0.4 blend proves the registered field is used.
    fvcc::DEShybrid<Scalar> des(
        exec, nfMesh, NeoN::Input(NeoN::TokenList({std::string("linear"), std::string("linearUpwind")}))
    );

    auto i1 = makeFilled<SurfScalar>(exec, nfMesh, "i1", Scalar(0));
    auto i2 = makeFilled<SurfScalar>(exec, nfMesh, "i2", Scalar(0));
    auto iDes = makeFilled<SurfScalar>(exec, nfMesh, "iDes", Scalar(0));
    lin.interpolate(flux, src, i1);
    lup.interpolate(flux, src, i2);
    des.interpolate(flux, src, iDes);

    auto i1H = i1.internalVector().copyToHost();
    auto i2H = i2.internalVector().copyToHost();
    auto iDesH = iDes.internalVector().copyToHost();
    for (NeoN::localIdx i = 0; i < nfMesh.nInternalFaces(); ++i)
    {
        const Scalar expected = (Scalar(1) - sigmaVal) * i1H.view()[i] + sigmaVal * i2H.view()[i];
        REQUIRE(iDesH.view()[i] == Catch::Approx(expected).margin(1e-12));
    }
}

// Validates parsing of the OpenFOAM DEShybrid coefficients from a fvSchemes div-scheme token list,
// where integer-looking values are tokenised as labels and decimals as scalars (mixed types).
TEST_CASE("readDEShybridCoefficients parses the OpenFOAM div-scheme spec")
{
    // Gauss DEShybrid linear linearUpwind grad(U) delta 0.65 30 2 0 1 1.0e-03 1.0
    NeoN::TokenList tokens;
    tokens.insert(std::string("Gauss"));
    tokens.insert(std::string("DEShybrid"));
    tokens.insert(std::string("linear"));
    tokens.insert(std::string("linearUpwind"));
    tokens.insert(std::string("grad(U)"));
    tokens.insert(std::string("delta"));
    tokens.insert(Scalar(0.65));       // CDES (scalar)
    tokens.insert(NeoN::label(30));    // U0 (label)
    tokens.insert(NeoN::label(2));     // L0 (label)
    tokens.insert(NeoN::label(0));     // sigmaMin (label)
    tokens.insert(NeoN::label(1));     // sigmaMax (label)
    tokens.insert(Scalar(1.0e-3));     // OmegaLim (scalar)
    tokens.insert(Scalar(1.0));        // nutLim (scalar)

    const nf::DEShybridCoefficients c = nf::readDEShybridCoefficients(tokens);
    REQUIRE(c.CDES == Catch::Approx(0.65));
    REQUIRE(c.U0 == Catch::Approx(30.0));
    REQUIRE(c.L0 == Catch::Approx(2.0));
    REQUIRE(c.sigmaMin == Catch::Approx(0.0));
    REQUIRE(c.sigmaMax == Catch::Approx(1.0));
    REQUIRE(c.OmegaLim == Catch::Approx(1.0e-3));
    REQUIRE(c.nutLim == Catch::Approx(1.0));

    // nutLim omitted -> defaults to 1.0
    NeoN::TokenList noNutLim;
    noNutLim.insert(std::string("DEShybrid"));
    noNutLim.insert(std::string("linear"));
    noNutLim.insert(std::string("linearUpwind"));
    noNutLim.insert(std::string("delta"));
    noNutLim.insert(Scalar(0.65));
    noNutLim.insert(NeoN::label(30));
    noNutLim.insert(NeoN::label(2));
    noNutLim.insert(NeoN::label(0));
    noNutLim.insert(NeoN::label(1));
    noNutLim.insert(Scalar(1.0e-3));
    const nf::DEShybridCoefficients c2 = nf::readDEShybridCoefficients(noNutLim);
    REQUIRE(c2.nutLim == Catch::Approx(1.0));
    REQUIRE(c2.OmegaLim == Catch::Approx(1.0e-3));
}

// Cross-checks the NeoFOAM blending factor against OpenFOAM's own DEShybrid::blendingFactor() on the
// same case, feeding the kernel OpenFOAM's exact fvc::grad(U), nut, nu and LES delta so the only
// thing under test is the sigma formula + cell->face interpolation.
TEST_CASE("DEShybrid sigma matches OpenFOAM's DEShybrid::blendingFactor")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto meshAdapter = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshAdapter; // also usable as a Foam::fvMesh
    const auto& nfMesh = mesh.nfMesh();

    // Make OpenFOAM's blended turbulence-model schemes (DEShybrid) available via runtime selection.
    // A by-name RTS use adds no link symbol, so load explicitly rather than rely on link order.
    runTime.libs().open(Foam::fileName("libturbulenceModelSchemes.so"));

    // --- OpenFOAM turbulence setup (LES SpalartAllmarasDDES, per the shared fixture) ---
    Foam::volVectorField U(
        Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::NO_WRITE),
        mesh
    );
    Foam::surfaceScalarField phi(
        Foam::IOobject("phi", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::NO_WRITE),
        Foam::fvc::flux(U)
    );
    Foam::singlePhaseTransportModel transport(U, phi);
    Foam::IOdictionary transportProperties(Foam::IOobject(
        "transportProperties",
        runTime.constant(),
        mesh,
        Foam::IOobject::MUST_READ_IF_MODIFIED,
        Foam::IOobject::NO_WRITE
    ));
    Foam::dimensionedScalar viscosity("nu", Foam::dimViscosity, transportProperties);

    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );
    foamTurb->validate();
    const Foam::volScalarField& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");
    const Foam::incompressible::LESModel& lesModel =
        Foam::refCast<const Foam::incompressible::LESModel>(foamTurb());
    const Foam::volScalarField& delta = lesModel.delta();

    // --- OpenFOAM reference sigma via DEShybrid::blendingFactor ---
    const std::string spec =
        "DEShybrid linear linearUpwind grad(U) " + delta.name() + " 0.05 30 2 0 1 1e-3 1.0";
    Foam::IStringStream is(spec);
    // Use the faceFlux overload: DEShybrid is a div scheme, so its sub-schemes (linearUpwind)
    // need the flux. The no-flux New would make linearUpwind read "grad(U)" as a flux field name.
    Foam::tmp<Foam::surfaceInterpolationScheme<Foam::vector>> tScheme(
        Foam::surfaceInterpolationScheme<Foam::vector>::New(mesh, phi, is)
    );
    const auto& blended = Foam::refCast<const Foam::blendedSchemeBase<Foam::vector>>(tScheme());
    Foam::tmp<Foam::surfaceScalarField> tOfSigma(blended.blendingFactor(U));
    const Foam::surfaceScalarField& ofSigma = tOfSigma();

    // --- NeoFOAM sigma fed OpenFOAM's exact inputs (isolates the formula) ---
    Foam::tmp<Foam::volTensorField> tGradU(Foam::fvc::grad(U));
    const Foam::volTensorField& ofGradU = tGradU();

    VolTensor nfGradU(
        exec, "gradU", nfMesh, fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Tensor>>(nfMesh)
    );
    nfGradU.internalVector() = NeoFOAM::fromFoamField(exec, ofGradU.primitiveField());

    auto [nfNut, nfDelta] = NeoFOAM::constFromMany(exec, nfMesh, ofNut, delta);

    VolScalar nfNu(exec, "nu", nfMesh, fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Scalar>>(nfMesh));
    fill(nfNu.internalVector(), viscosity.value());
    fill(nfNu.boundaryData().value(), viscosity.value());

    nf::DEShybridCoefficients coeffs;
    // CDES is reduced from the physical 0.65 so that, on this case's flow, sigma lands in the
    // tanh transition band (the formula's sensitive region) rather than saturating at sigmaMax.
    // Both OpenFOAM and NeoFOAM use the same value, so this still validates exact agreement.
    coeffs.CDES = 0.05;
    coeffs.U0 = 30.0;
    coeffs.L0 = 2.0;
    coeffs.sigmaMin = 0.0;
    coeffs.sigmaMax = 1.0;
    coeffs.OmegaLim = 1.0e-3;
    coeffs.nutLim = 1.0;

    auto surfInterp = fvcc::SurfaceInterpolation<Scalar>(
        exec, nfMesh, NeoN::Input(NeoN::TokenList({std::string("linear")}))
    );
    SurfScalar nfSigma(
        exec, "nfSigma", nfMesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<Scalar>>(nfMesh)
    );
    nf::computeDEShybridBlendingFactor(nfGradU, nfNut, nfNu, nfDelta, coeffs, surfInterp, nfSigma);

    // --- compare on internal faces (boundary-face interpolation differs by construction) ---
    auto nfH = nfSigma.internalVector().copyToHost();
    Scalar maxDiff = 0.0, sigMin = 1.0e30, sigMax = -1.0e30;
    for (NeoN::localIdx f = 0; f < nfMesh.nInternalFaces(); ++f)
    {
        const Scalar nfv = nfH.view()[f];
        const Scalar ofv = ofSigma.primitiveField()[static_cast<Foam::label>(f)];
        maxDiff = std::max(maxDiff, std::abs(nfv - ofv));
        sigMin = std::min(sigMin, ofv);
        sigMax = std::max(sigMax, ofv);
        REQUIRE(nfv == Catch::Approx(ofv).margin(1e-10));
    }
    INFO("max|nf-of| = " << maxDiff << ", OF sigma range = [" << sigMin << ", " << sigMax << "]");
    REQUIRE(maxDiff < 1e-10);
    // Ensure the comparison exercised the tanh transition rather than a saturated bound: sigma must
    // be strictly inside (sigmaMin, sigmaMax) somewhere, so neither all-at-floor nor all-at-ceiling.
    REQUIRE(sigMax > coeffs.sigmaMin + 1e-3);
    REQUIRE(sigMin < coeffs.sigmaMax - 1e-3);
}
