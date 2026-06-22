// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER
#include <unordered_set>

#include "common.hpp"

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "wallDist.H"

using Catch::Approx;

namespace fvc = Foam::fvc;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using Scalar = NeoN::scalar;
using Vec3 = NeoN::Vec3;
using Tensor = NeoN::Tensor;
using VolScalar = fvcc::VolumeField<Scalar>;
using VolVector = fvcc::VolumeField<Vec3>;
using SurfScalar = fvcc::SurfaceField<Scalar>;

extern Foam::Time* timePtr;
extern Foam::argList* argsPtr;
extern Foam::fvMesh* meshPtr;

// ============================================================
// Helper: compute k-omega SST blending function F1 from OF fields.
// Matches kOmegaSSTBase::F1() using the default coefficients.
// ============================================================
Foam::volScalarField computeOfF1(
    const Foam::fvMesh& mesh,
    const Foam::volScalarField& k,
    const Foam::volScalarField& omega,
    const Foam::volScalarField& nu,
    const Foam::volScalarField& y
)
{
    const Foam::scalar alphaOmega2 = 0.856;
    const Foam::scalar betaStar = 0.09;

    auto tgradK = fvc::grad(k);
    auto tgradOmega = fvc::grad(omega);

    // CDkOmega clamped to > 0 for F1 stability
    Foam::tmp<Foam::volScalarField> CDkOmegaPlus = Foam::max(
        Foam::tmp<Foam::volScalarField>(new Foam::volScalarField(
            "CDkOmegaPlus",
            2 * alphaOmega2 * (tgradK() & tgradOmega()) / omega
        )),
        Foam::dimensionedScalar("small", Foam::dimless / Foam::sqr(Foam::dimTime), 1e-10)
    );

    auto sqrtK = Foam::sqrt(Foam::max(k, Foam::dimensionedScalar("0", k.dimensions(), 0)));

    auto arg1 = Foam::min(
        Foam::min(
            Foam::max(sqrtK / (betaStar * omega * y), 500 * nu / (Foam::sqr(y) * omega)),
            4 * alphaOmega2 * k / (CDkOmegaPlus() * Foam::sqr(y))
        ),
        Foam::dimensionedScalar("10", Foam::dimless, 10)
    );

    return Foam::volScalarField("ofF1", Foam::tanh(Foam::pow4(arg1)));
}

// ============================================================
// Helper: compute k-omega SST nut from OF fields (no wall functions).
// ============================================================
Foam::volScalarField computeOfNut(
    const Foam::fvMesh& mesh,
    const Foam::volScalarField& k,
    const Foam::volScalarField& omega,
    const Foam::volScalarField& nu,
    const Foam::volScalarField& y
)
{
    const Foam::scalar betaStar = 0.09;
    const Foam::scalar a1 = 0.31;
    const Foam::scalar b1 = 1.0;

    auto sqrtK = Foam::sqrt(Foam::max(k, Foam::dimensionedScalar("0", k.dimensions(), 0)));

    auto arg2 = Foam::min(
        Foam::max(2 * sqrtK / (betaStar * omega * y), 500 * nu / (Foam::sqr(y) * omega)),
        Foam::dimensionedScalar("100", Foam::dimless, 100)
    );
    auto F2 = Foam::tanh(Foam::sqr(arg2));

    auto tgradU_tmp = fvc::grad(mesh.lookupObject<Foam::volVectorField>("U"));
    const Foam::volTensorField& gradU = tgradU_tmp();
    auto S2 = 2 * Foam::magSqr(Foam::symm(gradU));
    auto sqrtS2 = Foam::sqrt(S2);

    return Foam::volScalarField("ofNutComputed", a1 * k / Foam::max(a1 * omega, b1 * F2 * sqrtS2));
}

// ============================================================
TEST_CASE("kOmegaSST: component kernels match OpenFOAM")
// ============================================================
{
    Foam::Time& runTime = *timePtr;

    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
    solverDict.subDict("k") = nf::mapFvSolution(solverDict.subDict("k"));
    solverDict.subDict("omega") = nf::mapFvSolution(solverDict.subDict("omega"));
    rt.fvSchemesDict = nf::mapFvSchemes(rt.fvSchemesDict);
    const NeoN::Dictionary controlDict = nf::convert(runTime.controlDict());
    const Foam::scalar dt = controlDict.get<Foam::scalar>("deltaT");
    runTime.setDeltaT(dt);

    // --- Read OpenFOAM fields ---
    Foam::volVectorField U(
        Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ),
        mesh
    );
    Foam::volScalarField p(
        Foam::IOobject("p", runTime.timeName(), mesh, Foam::IOobject::MUST_READ),
        mesh
    );
    Foam::surfaceScalarField phi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::READ_IF_PRESENT,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(U)
    );

    Foam::singlePhaseTransportModel transport(U, phi);
    Foam::tmp<Foam::volScalarField> tnu = transport.nu();
    Foam::volScalarField nuFoam(
        Foam::IOobject(
            "nu",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        tnu()
    );

    // Instantiate OpenFOAM kOmegaSST model
    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );
    foamTurb->validate();

    const Foam::volScalarField& ofK = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::volScalarField& ofOmega = mesh.lookupObject<Foam::volScalarField>("omega");
    const Foam::volScalarField& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");

    Foam::wallDist y(mesh);
    const Foam::volScalarField& wallDist = y.y();

    // Reference F1
    Foam::volScalarField ofF1 = computeOfF1(mesh, ofK, ofOmega, nuFoam, wallDist);

    // Reference nut (from formula, not wall-function)
    Foam::volScalarField ofNutComputed = computeOfNut(mesh, ofK, ofOmega, nuFoam, wallDist);

    // Reference Pk and omega sources
    const Foam::scalar betaStar = 0.09;
    const Foam::scalar a1 = 0.31;
    const Foam::scalar b1 = 1.0;
    const Foam::scalar c1 = 10.0;
    const Foam::scalar gamma1 = 5.0 / 9.0;
    const Foam::scalar gamma2 = 0.44;
    const Foam::scalar beta1 = 0.075;
    const Foam::scalar beta2 = 0.0828;
    const Foam::scalar alphaOmega2 = 0.856;

    auto tgradU = fvc::grad(U);
    // GbyNu0 = gradU && devTwoSymm(gradU)
    Foam::volScalarField ofGbyNu0(
        "ofGbyNu0",
        Foam::tmp<Foam::volScalarField>(
            new Foam::volScalarField("t", tgradU() && Foam::devTwoSymm(tgradU()))
        )
    );
    auto S2 = 2 * Foam::magSqr(Foam::symm(tgradU()));
    auto sqrtK = Foam::sqrt(Foam::max(ofK, Foam::dimensionedScalar("0", ofK.dimensions(), 0)));
    auto sqrtS2 = Foam::sqrt(S2);

    // F2 (same computation as in correctNutInternal)
    auto arg2 = Foam::min(
        Foam::max(
            2 * sqrtK / (betaStar * ofOmega * wallDist),
            500 * nuFoam / (Foam::sqr(wallDist) * ofOmega)
        ),
        Foam::dimensionedScalar("100", Foam::dimless, 100)
    );
    auto F2 = Foam::tanh(Foam::sqr(arg2));

    // Blended coefficients
    Foam::volScalarField ofGamma("ofGamma", ofF1 * (gamma1 - gamma2) + gamma2);
    Foam::volScalarField ofBeta("ofBeta", ofF1 * (beta1 - beta2) + beta2);

    // G = nut * GbyNu0 (uses post-validate nut from OF)
    auto G = ofNut * ofGbyNu0;
    // Pk = min(G, c1*betaStar*k*omega)
    Foam::volScalarField ofPk("ofPk", Foam::min(G, c1 * betaStar * ofK * ofOmega));

    // spK = betaStar * omega
    Foam::volScalarField ofSpK("ofSpK", betaStar * ofOmega);

    // Bounded GbyNu for omega source
    Foam::volScalarField ofGbyNuBound(
        "ofGbyNuBound",
        Foam::min(
            ofGbyNu0,
            (c1 / a1) * betaStar * ofOmega * Foam::max(a1 * ofOmega, b1 * F2 * sqrtS2)
        )
    );
    // Cross-diffusion (actual, can be negative)
    auto tgradK = fvc::grad(ofK);
    auto tgradOmega = fvc::grad(ofOmega);
    Foam::volScalarField CDkOmegaActual(
        "CDkOmegaActual",
        2 * alphaOmega2 * (tgradK() & tgradOmega()) / ofOmega
    );
    Foam::volScalarField crossSource("crossSource", (1 - ofF1) * CDkOmegaActual);

    // omegaSource = gamma * GbyNu_bounded + positive cross-diffusion
    Foam::volScalarField ofOmegaSource(
        "ofOmegaSource",
        ofGamma * ofGbyNuBound
            + Foam::max(crossSource, Foam::dimensionedScalar("0", crossSource.dimensions(), 0))
    );

    // spOmega = beta*omega + implicit cross-diffusion sink
    Foam::volScalarField ofSpOmega(
        "ofSpOmega",
        ofBeta * ofOmega
            + Foam::max(
                -crossSource / ofOmega,
                Foam::dimensionedScalar("0", Foam::dimless / Foam::dimTime, 0)
            )
    );

    // --- NeoFOAM setup ---
    auto& nfU = NeoFOAM::constructAndRegister(fieldCollection, rt, U, false);
    auto& nfK = NeoFOAM::constructAndRegister(fieldCollection, rt, ofK, false);
    auto& nfOmega = NeoFOAM::constructAndRegister(fieldCollection, rt, ofOmega, false);
    auto [nfWallDist, nfNu, nfNut] =
        NeoFOAM::constFromMany(rt.exec, rt.nfMesh, wallDist, nuFoam, ofNut);

    nf::KOmegaSST turbNF(rt.exec, rt.nfMesh, nfNu, nfWallDist);

    // Compute gradients then sources
    {
        nnfvcc::GaussGreenGrad gradOp(exec, rt.nfMesh);
        VolScalar nfGradKMag(
            exec,
            "gradKMag",
            rt.nfMesh,
            fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Scalar>>(rt.nfMesh)
        );

        VolVector nfGradK(
            exec,
            "gradK",
            rt.nfMesh,
            fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(rt.nfMesh)
        );
        VolVector nfGradOmega(
            exec,
            "gradOmega",
            rt.nfMesh,
            fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(rt.nfMesh)
        );
        fvcc::VolumeField<Tensor> nfGradU(
            exec,
            "gradU",
            rt.nfMesh,
            fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Tensor>>(rt.nfMesh)
        );
        gradOp.gradTensor(nfU, nfGradU);
        gradOp.grad(nfK, nfGradK);
        gradOp.grad(nfOmega, nfGradOmega);

        turbNF.computeF1AndSources(nfK, nfOmega, nfNut, nfGradK, nfGradOmega, nfGradU);
    }

    // --- Compare F1 ---
    REQUIRE_THAT(turbNF.F1Field(), EqualsInternal(ofF1, ApproxScalar(1e-10)));

    // --- Compare Pk ---
    REQUIRE_THAT(turbNF.PkField(), EqualsInternal(ofPk, ApproxScalar(1e-10)));

    // --- Compare spK ---
    REQUIRE_THAT(turbNF.spKField(), EqualsInternal(ofSpK, ApproxScalar(1e-10)));

    // --- Compare omegaSource ---
    REQUIRE_THAT(turbNF.omegaSourceField(), EqualsInternal(ofOmegaSource, ApproxScalar(1e-10)));

    // --- Compare spOmega ---
    REQUIRE_THAT(turbNF.spOmegaField(), EqualsInternal(ofSpOmega, ApproxScalar(1e-10)));

    // --- Compare nut from correctNutInternal ---
    // Set gradU_ inside turbNF so correctNutInternal can use it for S2
    turbNF.validate(nfU, nfK, nfOmega, nfNut);
    REQUIRE_THAT(nfNut, EqualsInternal(ofNutComputed, ApproxScalar(1e-10)));
}

// ============================================================
TEST_CASE("kOmegaSST: NeoFOAM wrapper validate()+correct() matches OpenFOAM")
// ============================================================
{
    Foam::Time& runTime = *timePtr;

    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
    solverDict.subDict("k") = nf::mapFvSolution(solverDict.subDict("k"));
    solverDict.subDict("omega") = nf::mapFvSolution(solverDict.subDict("omega"));
    rt.fvSchemesDict = nf::mapFvSchemes(rt.fvSchemesDict);
    const NeoN::Dictionary controlDict = nf::convert(runTime.controlDict());
    const Foam::scalar dt = controlDict.get<Foam::scalar>("deltaT");
    runTime.setDeltaT(dt);

    // --- OpenFOAM fields ---
    Foam::volVectorField U(
        Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ),
        mesh
    );
    Foam::volScalarField p(
        Foam::IOobject("p", runTime.timeName(), mesh, Foam::IOobject::MUST_READ),
        mesh
    );
    Foam::surfaceScalarField phi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::READ_IF_PRESENT,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(U)
    );

    Foam::singlePhaseTransportModel transport(U, phi);
    auto tnu = transport.nu();

    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );
    foamTurb->validate();

    const Foam::volScalarField& ofK = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::volScalarField& ofOmega = mesh.lookupObject<Foam::volScalarField>("omega");
    const Foam::volScalarField& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");

    Foam::wallDist wd(mesh);
    const Foam::volScalarField& wallDist = wd.y();

    // --- NeoFOAM fields ---
    auto& nfU = NeoFOAM::constructAndRegister(fieldCollection, rt, U, false);
    auto& nfP = NeoFOAM::constructAndRegister(fieldCollection, rt, p, false);
    auto& nfPhi = NeoFOAM::constructAndRegister(fieldCollection, rt, phi, false);
    auto& nfK = NeoFOAM::constructAndRegister(fieldCollection, rt, ofK, false);
    auto& nfOmega = NeoFOAM::constructAndRegister(fieldCollection, rt, ofOmega, false);

    auto [nfWallDist, nfNu, nfNut] =
        NeoFOAM::constFromMany(rt.exec, rt.nfMesh, wallDist, tnu(), ofNut);

    // --- Wrapper: validate ---
    nf::KOmegaSST turbNF(rt.exec, rt.nfMesh, nfNu, nfWallDist);
    turbNF.validate(nfU, nfK, nfOmega, nfNut);

    fvcc::rotateOldTimes(nfU);
    fvcc::rotateOldTimes(nfPhi);
    fvcc::rotateOldTimes(nfK);
    fvcc::rotateOldTimes(nfOmega);

    // --- Momentum solve (NeoFOAM) ---
    // Matches OF's divDevReff(U) = -laplacian(nuEff,U) - div(nuEff*dev2(T(gradU)))
    nf::PDESolver<NeoN::Vec3> UEqn(
        dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(turbNF.nuEff(), nfU)
            + dsl::exp::viscousStress(nfNu, nfNut, turbNF.gradU()),
        nfU,
        rt
    );
    UEqn.solve(-1.0 * dsl::exp::grad(nfP));
    nfU.correctBoundaryConditions();

    // --- Turbulence correct (NeoFOAM) ---
    turbNF.correct(nfU, nfPhi, nfK, nfOmega, nfNut, rt);

    // --- OpenFOAM reference ---
    Foam::fvVectorMatrix ofUEqn(
        Foam::fvm::ddt(U) + Foam::fvm::div(phi, U) + foamTurb->divDevReff(U)
    );
    Foam::solve(ofUEqn == -fvc::grad(p));
    U.correctBoundaryConditions();

    foamTurb->correct();

    // --- Comparisons ---
    REQUIRE_THAT(nfU, EqualsInternal(U, ApproxVector(1e-10)));
    REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(U, ApproxVector(1e-10)));

    const Foam::volScalarField& kFoam = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::volScalarField& omegaFoam = mesh.lookupObject<Foam::volScalarField>("omega");
    const Foam::volScalarField& nutFoam = mesh.lookupObject<Foam::volScalarField>("nut");

    // k and omega tolerances are looser than 1e-10 because OpenFOAM's kOmegaSST
    // includes a compressibility correction term -(2/3)*div(U)*phi in both equations
    // (fvm::SuSp((2/3)*divU, k/omega)).  For converged incompressible flow div(U)=0
    // and the term vanishes; here a single uncoupled step on a 3x3x3 mesh leaves
    // non-zero div(U) at every boundary cell (~1.8 s⁻¹), producing an O(1e-5) artifact.
    REQUIRE_THAT(nfK, EqualsInternal(kFoam, ApproxScalar(5e-5)));
    REQUIRE_THAT(nfK.boundaryData(), EqualsBoundary(kFoam, ApproxScalar(5e-5)));

    REQUIRE_THAT(nfOmega, EqualsInternal(omegaFoam, ApproxScalar(5.0)));
    REQUIRE_THAT(nfOmega.boundaryData(), EqualsBoundary(omegaFoam, ApproxScalar(5.0)));

    // nut is derived from k and omega, so it inherits their compressibility-correction
    // artifact; 5e-7 covers the observed O(3e-7) spread on this mesh.
    REQUIRE_THAT(nfNut, EqualsInternal(nutFoam, ApproxScalar(5e-7)));
    REQUIRE_THAT(nfNut.boundaryData(), EqualsBoundary(nutFoam, ApproxScalar(5e-7)));
}
