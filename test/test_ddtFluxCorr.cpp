// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"
#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "constrainHbyA.H"

using Catch::Approx;

namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;
namespace fvcc = NeoN::finiteVolume::cellCentred;

using Scalar = NeoN::scalar;
using Vec3 = NeoN::Vec3;
using VolScalar = fvcc::VolumeField<Scalar>;
using VolVector = fvcc::VolumeField<Vec3>;
using SurfScalar = fvcc::SurfaceField<Scalar>;

extern Foam::Time* timePtr;

TEST_CASE("ddtCorr: OpenFOAM Euler vs NeoN (BDF1)")
{
    Foam::Time& runTime = *timePtr;
    // --- NeoN database / collection
    NeoN::Database db;
    auto& fieldCollection = fvcc::VectorCollection::instance(db, "fieldCollection");

    // const Foam::word execName = "GPU";
    // auto exec = NeoN::GPUExecutor();
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // --- FOAM time
    const Foam::scalar startTime = 0.0;
    const Foam::label startTimeIndex = 0;
    runTime.setTime(startTime, startTimeIndex);

    // --- Mesh
    auto rt = NeoFOAM::createAdapterRunTime(runTime,exec);
    auto& mesh = rt.mesh;
    //std::unique_ptr<NeoFOAM::MeshAdapter> meshAdapterPtr = NeoFOAM::createMesh(exec, runTime);
    //NeoFOAM::MeshAdapter& mesh = *meshAdapterPtr;
    NeoN::UnstructuredMesh& nfMesh = mesh.nfMesh();

    // --- Time step
    const NeoN::Dictionary controlDict = NeoFOAM::convert(runTime.controlDict());
    const Foam::scalar dt = controlDict.get<Foam::scalar>("deltaT");

    runTime.setDeltaT(dt);

    // --- FOAM fields
    Foam::volVectorField U(
        Foam::IOobject(
            "U",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh
    );

    Foam::volScalarField p
    (
        Foam::IOobject
        (
            "p",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh
    );


    Foam::surfaceScalarField phi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(U)
    );

    // --- OpenFOAM reference
    Foam::surfaceScalarField foamCorr(fvc::ddtCorr(U, phi)());

    // === NeoN: mirror state ===

    auto& nfU = fieldCollection.registerVector<VolVector>(
        NeoFOAM::CreateFromFoamField<Foam::volVectorField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = U,
            .name = "nfU"
        }
    );

     auto& nfp = fieldCollection.registerVector<VolScalar>(
        NeoFOAM::CreateFromFoamField<Foam::volScalarField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = p,
            .name = "nfp"
        }
    );

    auto& nfPhi = fieldCollection.registerVector<SurfScalar>(
        NeoFOAM::CreateFromFoamField<Foam::surfaceScalarField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = phi,
            .name = "nfPhi"
        }
    );

    auto& nfU0 = fvcc::oldTime(nfU);
    auto& nfPhi0 = fvcc::oldTime(nfPhi);

    NeoN::Dictionary fvSchemes, ddtSchemes, timeIntegrationDict;
    // timeIntegrationDict.insert("type", std::string("backwardEuler")); // Euler
    ddtSchemes.insert("ddt(nfU)", std::string("BDF1")); // Euler
    // fvSchemes.insert("timeIntegration", timeIntegrationDict);
    fvSchemes.insert("ddtSchemes", ddtSchemes);

    // --- DdtOperator for momentum
    fvcc::DdtOperator<Vec3> ddtOp(NeoN::dsl::Operator::Type::Implicit, nfU);
    ddtOp.read(fvSchemes);

    const auto& scheme = ddtOp.scheme();

    // --- Compute NeoN correction
    SurfScalar nfCorr = scheme.ddtFluxCorr(nfU, nfPhi, dt);

    // --- Sanity: states match
    NeoFOAM::compare(nfU, U, ApproxVector(1e-15));
    NeoFOAM::compare(nfU0, U.oldTime(), ApproxVector(1e-15));
    NeoFOAM::compare(nfPhi, phi, ApproxScalar(1e-15));
    NeoFOAM::compare(nfPhi0, phi.oldTime(), ApproxScalar(1e-15));

    SECTION("ddtCorr " + execName) { NeoFOAM::compare(nfCorr, foamCorr, ApproxScalar(1e-15)); }

    Foam::dimensionedScalar nu
    (
        "nu",
        Foam::dimViscosity,
        0.01
    );

    Foam::fvVectorMatrix UEqn
    (
        fvm::ddt(U)
      + fvm::div(phi, U)
      - fvm::laplacian(nu, U)
    );

    Foam::volScalarField rAU(1.0/UEqn.A());
    Foam::volVectorField HbyA(Foam::constrainHbyA(rAU*UEqn.H(), U, p));
    Foam::surfaceScalarField phiHbyA
    (
        "phiHbyA",
        fvc::flux(HbyA)
       +fvc::interpolate(rAU)*fvc::ddtCorr(U, phi)
    );

    Foam::fvScalarMatrix pEqnOF
    (
        fvm::laplacian(rAU, p) == fvc::div(phiHbyA)
    );
    pEqnOF.setReference(int(0), scalar(0));
    pEqnOF.solve();

    auto nuBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh);
    SurfScalar nuNF(exec, "nu", nfMesh, nuBCs);
    NeoN::fill(nuNF.internalVector(), scalar(0.01));
    NeoN::fill(nuNF.boundaryData().value(), scalar(0.01));
    NeoFOAM::PDESolver<NeoN::Vec3> UEqnNF(
        NeoN::dsl::ddt(nfU) + NeoN::dsl::imp::div(nfPhi, nfU) - NeoN::dsl::imp::laplacian(nuNF, nfU),
        nfU,
        rt
    );

    UEqnNF.assemble();
    auto [crAU, hByANF] = NeoFOAM::computeRAUandHByA(UEqnNF);
    NeoFOAM::constrainHbyA(nfU, nfp, hByANF);

    SurfScalar rAUNF =
        fvcc::SurfaceInterpolation<NeoN::scalar>(
            rt.exec,
            rt.nfMesh,
            NeoN::TokenList({std::string("linear")})
        )
        .interpolate(crAU);
    rAUNF.name = "rAUfNF";

    auto phiHbyANF = NeoFOAM::flux(hByANF);

    NeoFOAM::PDESolver<NeoN::scalar> pEqnNF(
        NeoN::dsl::imp::laplacian(rAUNF, nfp) - NeoN::dsl::exp::div(phiHbyANF),
        nfp,
        rt
    );
    pEqnNF.setReference(NeoN::localIdx(0), NeoN::scalar(0));
    const auto* ddtScheme = UEqnNF.ddtScheme();

    pEqnNF.enableDdtFluxCorr(*ddtScheme, nfU, nfPhi);

    auto stats = pEqnNF.solve();
    nfp.correctBoundaryConditions();

    SECTION("application to actual fields: " + execName)
    {
        NeoFOAM::compare(nfp, p, ApproxScalar(1e-5));
    }
}
