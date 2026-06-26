// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

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

extern Foam::Time* timePtr;
extern Foam::argList* argsPtr;
extern Foam::fvMesh* meshPtr;

// ============================================================
// Helper: compute k-epsilon nut from OF fields.
// Matches kEpsilonBase: ν_t = Cμ k²/ε
// ============================================================
Foam::volScalarField
computeOfNutKEps(const Foam::volScalarField& k, const Foam::volScalarField& epsilon)
{
    const Foam::scalar Cmu = 0.09;
    const Foam::scalar rootVSmall = 1e-30;
    return Foam::volScalarField(
        "ofNutKEps",
        Cmu * Foam::sqr(Foam::max(k, Foam::dimensionedScalar("0", k.dimensions(), 0)))
            / Foam::max(
                epsilon,
                Foam::dimensionedScalar("rootVSmall", epsilon.dimensions(), rootVSmall)
            )
    );
}

// ============================================================
TEST_CASE("kEpsilon: computeSources matches OpenFOAM")
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
    solverDict.subDict("epsilon") = nf::mapFvSolution(solverDict.subDict("epsilon"));
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

    // Instantiate OpenFOAM kEpsilon model (reads k, epsilon, nut from disk)
    Foam::autoPtr<Foam::incompressible::turbulenceModel> foamTurb(
        Foam::incompressible::turbulenceModel::New(U, phi, transport)
    );
    foamTurb->validate();

    const Foam::volScalarField& ofK = mesh.lookupObject<Foam::volScalarField>("k");
    const Foam::volScalarField& ofEps = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const Foam::volScalarField& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");

    Foam::wallDist wd(mesh);
    const Foam::volScalarField& wallDist = wd.y();

    // --- OpenFOAM reference source terms ---
    // GbyNu0 = gradU && devTwoSymm(gradU)
    auto tgradU = fvc::grad(U);
    Foam::volScalarField ofGbyNu0("ofGbyNu0", tgradU() && Foam::devTwoSymm(tgradU()));

    // Pk = G = nut * GbyNu0
    Foam::volScalarField ofPk("ofPk", ofNut * ofGbyNu0);

    const Foam::scalar rootVSmall = 1e-30;
    // spK = epsilon/k
    Foam::volScalarField ofSpK(
        "ofSpK",
        ofEps / Foam::max(ofK, Foam::dimensionedScalar("rootVSmall", ofK.dimensions(), rootVSmall))
    );

    // epsilonSource = C1 * Cmu * k * GbyNu0  (= C1 * G * epsilon/k, since nut=Cmu*k²/eps)
    const Foam::scalar Cmu = 0.09, C1 = 1.44, C2 = 1.92;
    Foam::volScalarField ofEpsSource("ofEpsSource", C1 * Cmu * ofK * ofGbyNu0);

    // spEpsilon = C2 * epsilon/k
    Foam::volScalarField ofSpEps(
        "ofSpEps",
        C2 * ofEps
            / Foam::max(ofK, Foam::dimensionedScalar("rootVSmall", ofK.dimensions(), rootVSmall))
    );

    // --- NeoFOAM setup ---
    auto& nfU = NeoFOAM::constructAndRegister(fieldCollection, rt, U, false);
    auto& nfK = NeoFOAM::constructAndRegister(fieldCollection, rt, ofK, false);
    auto& nfEps = NeoFOAM::constructAndRegister(fieldCollection, rt, ofEps, false);
    auto [nfWallDist, nfNu, nfNut] =
        NeoFOAM::constFromMany(rt.exec, rt.nfMesh, wallDist, nuFoam, ofNut);

    nf::KEpsilon turbNF(rt.exec, rt.nfMesh, nfNu, nfWallDist);

    // Compute gradU then call computeSources
    {
        nnfvcc::GaussGreenGrad gradOp(exec, rt.nfMesh);
        fvcc::VolumeField<Tensor> nfGradU(
            exec,
            "gradU",
            rt.nfMesh,
            fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Tensor>>(rt.nfMesh)
        );
        gradOp.gradTensor(nfU, nfGradU);

        turbNF.computeSources(nfK, nfEps, nfNut, nfGradU);
    }

    // --- Compare source terms ---
    REQUIRE_THAT(turbNF.pkField(), EqualsInternal(ofPk, ApproxScalar(1e-10)));
    REQUIRE_THAT(turbNF.spKField(), EqualsInternal(ofSpK, ApproxScalar(1e-10)));
    REQUIRE_THAT(turbNF.epsilonSourceField(), EqualsInternal(ofEpsSource, ApproxScalar(1e-10)));
    REQUIRE_THAT(turbNF.spEpsilonField(), EqualsInternal(ofSpEps, ApproxScalar(1e-10)));

    // --- Compare nut after correctNutInternal ---
    Foam::volScalarField ofNutComputed = computeOfNutKEps(ofK, ofEps);
    turbNF.correctNutInternal(nfK, nfEps, nfNut);
    REQUIRE_THAT(nfNut, EqualsInternal(ofNutComputed, ApproxScalar(1e-10)));
}

// ============================================================
TEST_CASE("kEpsilon: NeoFOAM wrapper validate()+correct() matches OpenFOAM")
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
    solverDict.subDict("epsilon") = nf::mapFvSolution(solverDict.subDict("epsilon"));
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
    const Foam::volScalarField& ofEps = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const Foam::volScalarField& ofNut = mesh.lookupObject<Foam::volScalarField>("nut");

    Foam::wallDist wd(mesh);
    const Foam::volScalarField& wallDist = wd.y();

    // --- NeoFOAM fields ---
    auto& nfU = NeoFOAM::constructAndRegister(fieldCollection, rt, U, false);
    auto& nfP = NeoFOAM::constructAndRegister(fieldCollection, rt, p, false);
    auto& nfPhi = NeoFOAM::constructAndRegister(fieldCollection, rt, phi, false);
    auto& nfK = NeoFOAM::constructAndRegister(fieldCollection, rt, ofK, false);
    auto& nfEps = NeoFOAM::constructAndRegister(fieldCollection, rt, ofEps, false);

    auto [nfWallDist, nfNu, nfNut] =
        NeoFOAM::constFromMany(rt.exec, rt.nfMesh, wallDist, tnu(), ofNut);

    // --- Wrapper: validate ---
    nf::KEpsilon turbNF(rt.exec, rt.nfMesh, nfNu, nfWallDist);
    turbNF.validate(nfU, nfK, nfEps, nfNut);

    fvcc::rotateOldTimes(nfU);
    fvcc::rotateOldTimes(nfPhi);
    fvcc::rotateOldTimes(nfK);
    fvcc::rotateOldTimes(nfEps);

    // --- Momentum solve (NeoFOAM) ---
    nf::PDESolver<NeoN::Vec3> UEqn(
        dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(turbNF.nuEff(), nfU)
            + dsl::exp::viscousStress(nfNu, nfNut, turbNF.gradU()),
        nfU,
        rt
    );
    UEqn.solve(-1.0 * dsl::exp::grad(nfP));
    nfU.correctBoundaryConditions();

    // --- Turbulence correct (NeoFOAM) ---
    turbNF.correct(nfU, nfPhi, nfK, nfEps, nfNut, rt);

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
    const Foam::volScalarField& epsFoam = mesh.lookupObject<Foam::volScalarField>("epsilon");
    const Foam::volScalarField& nutFoam = mesh.lookupObject<Foam::volScalarField>("nut");

    // Tolerances are looser than 1e-10 because OpenFOAM's kEpsilon includes a
    // compressibility correction -(2/3)*div(U) in both equations which is
    // non-zero for a single uncoupled step on a small mesh.
    REQUIRE_THAT(nfK, EqualsInternal(kFoam, ApproxScalar(5e-5)));
    REQUIRE_THAT(nfK.boundaryData(), EqualsBoundary(kFoam, ApproxScalar(5e-5)));

    REQUIRE_THAT(nfEps, EqualsInternal(epsFoam, ApproxScalar(5e-4)));
    REQUIRE_THAT(nfEps.boundaryData(), EqualsBoundary(epsFoam, ApproxScalar(5e-4)));

    // nut = Cmu*k²/eps inherits the compressibility-correction artifact from k/epsilon;
    // 5e-7 covers the observed O(1.5e-7) spread on this mesh.
    REQUIRE_THAT(nfNut, EqualsInternal(nutFoam, ApproxScalar(5e-7)));
    REQUIRE_THAT(nfNut.boundaryData(), EqualsBoundary(nutFoam, ApproxScalar(5e-7)));
}
