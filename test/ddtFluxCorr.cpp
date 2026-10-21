// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"
#include "NeoN/NeoN.hpp"
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
    float epsilon = 1e-15;
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
    auto rt = NeoFOAM::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
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

    Foam::volScalarField p(
        Foam::IOobject(
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

    auto& nfU = NeoFOAM::constructAndRegister(fieldCollection, rt, U);
    auto& nfP = NeoFOAM::constructAndRegister(fieldCollection, rt, p, false);
    auto& nfPhi = NeoFOAM::constructAndRegister(fieldCollection, rt, phi);

    auto& nfU0 = fvcc::oldTime(nfU);
    auto& nfPhi0 = fvcc::oldTime(nfPhi);

    NeoN::Dictionary ddtSchemes;
    ddtSchemes.insert("ddt(U)", std::string("BDF1")); // Euler
    rt.fvSchemesDict.insert("ddtSchemes", ddtSchemes);

    // --- DdtOperator for momentum
    fvcc::DdtOperator<Vec3> ddtOp(NeoN::dsl::Operator::Type::Implicit, nfU);
    ddtOp.read(rt.fvSchemesDict);

    auto scheme = ddtOp.scheme();

    // --- Compute NeoN correction
    SurfScalar nfCorr = ddtFluxCorr(nfU, nfPhi, dt, scheme);

    // --- Sanity: states match
    REQUIRE_THAT(nfU, EqualsInternal(U, ApproxVector(epsilon)));
    REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(U, ApproxVector(epsilon)));
    REQUIRE_THAT(nfU0, EqualsInternal(U.oldTime(), ApproxVector(epsilon)));
    REQUIRE_THAT(nfU0.boundaryData(), EqualsBoundary(U.oldTime(), ApproxVector(epsilon)));
    REQUIRE_THAT(nfPhi, EqualsInternal(phi, ApproxScalar(epsilon)));
    REQUIRE_THAT(nfPhi.boundaryData(), EqualsBoundary(phi, ApproxScalar(epsilon)));
    REQUIRE_THAT(nfPhi0, EqualsInternal(phi.oldTime(), ApproxScalar(epsilon)));
    REQUIRE_THAT(nfPhi0.boundaryData(), EqualsBoundary(phi.oldTime(), ApproxScalar(epsilon)));

    SECTION("ddtCorr " + execName)
    {
        REQUIRE_THAT(nfCorr, EqualsInternal(foamCorr, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfCorr.boundaryData(), EqualsBoundary(foamCorr, ApproxScalar(epsilon)));
    }

    Foam::dimensionedScalar nu("nu", Foam::dimViscosity, 0.01);

    Foam::fvVectorMatrix UEqn(fvm::ddt(U) + fvm::div(phi, U) - fvm::laplacian(nu, U));

    Foam::volScalarField rAU(1.0 / UEqn.A());
    Foam::volVectorField HbyA(Foam::constrainHbyA(rAU * UEqn.H(), U, p));
    Foam::surfaceScalarField phiHbyA(
        "phiHbyA",
        fvc::flux(HbyA) + fvc::interpolate(rAU) * fvc::ddtCorr(U, phi)
    );

    auto nuBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh);
    SurfScalar nuNF(exec, "nu", nfMesh, nuBCs);
    NeoN::fill(nuNF.internalVector(), scalar(0.01));
    NeoN::fill(nuNF.boundaryData().value(), scalar(0.01));
    NeoFOAM::PDE<NeoN::Vec3> UEqnNF(
        NeoN::dsl::imp::ddt(nfU) + NeoN::dsl::imp::div(nfPhi, nfU)
            - NeoN::dsl::imp::laplacian(nuNF, nfU),
        nfU,
        rt
    );

    UEqnNF.assemble();
    auto [crAU, hByAND] = NeoFOAM::computeRAUandHByA(UEqnNF);
    NeoFOAM::constrainHbyA(nfU, nfP, hByAND);

    SurfScalar rAUNF = fvcc::SurfaceInterpolation<NeoN::scalar>(
                           rt.exec,
                           rt.nfMesh,
                           NeoN::TokenList({std::string("linear")})
    )
                           .interpolate(crAU);
    rAUNF.name = "rAUfNF";

    auto phiHbyAND = NeoFOAM::flux(hByAND) + nfCorr * rAUNF;
    SECTION("application to actual fields: " + execName)
    {
        REQUIRE_THAT(phiHbyAND, EqualsInternal(phiHbyA, ApproxScalar(1e-15)));
    }
}
