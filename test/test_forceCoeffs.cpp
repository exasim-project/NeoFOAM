// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "NeoFOAM/functionObjects/forces.hpp"
#include "NeoFOAM/functionObjects/forceCoeffs.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"

namespace nf = NeoFOAM;
namespace fvcc = NeoN::finiteVolume::cellCentred;

extern Foam::Time* timePtr;


// ── helpers ──────────────────────────────────────────────────────────────────

/**
 * Compute the pressure force on a single patch using plain OpenFOAM arithmetic.
 * This is the reference against which NeoFOAM::Forces is compared.
 */
Foam::vector ofPressureForce(
    const Foam::volScalarField& p,
    const Foam::fvMesh& mesh,
    const Foam::word& patchName,
    Foam::scalar rhoRef,
    Foam::scalar pRef
)
{
    const Foam::label patchi = mesh.boundaryMesh().findPatchID(patchName);
    REQUIRE(patchi >= 0);

    const auto& pBc = p.boundaryField()[patchi];
    const Foam::vectorField& Sf = mesh.boundary()[patchi].Sf();

    Foam::vector fp = Foam::vector::zero;
    forAll(pBc, facei)
    {
        fp += rhoRef * (pBc[facei] - pRef) * Sf[facei];
    }
    return fp;
}

/**
 * Compute the pressure moment about a point on a single patch.
 */
Foam::vector ofPressureMoment(
    const Foam::volScalarField& p,
    const Foam::fvMesh& mesh,
    const Foam::word& patchName,
    Foam::scalar rhoRef,
    Foam::scalar pRef,
    const Foam::vector& cofR
)
{
    const Foam::label patchi = mesh.boundaryMesh().findPatchID(patchName);
    REQUIRE(patchi >= 0);

    const auto& pBc = p.boundaryField()[patchi];
    const Foam::vectorField& Sf = mesh.boundary()[patchi].Sf();
    const Foam::vectorField& Cf = mesh.boundary()[patchi].Cf();

    Foam::vector mp = Foam::vector::zero;
    forAll(pBc, facei)
    {
        Foam::vector fp = rhoRef * (pBc[facei] - pRef) * Sf[facei];
        Foam::vector lv = Cf[facei] - cofR;
        mp += lv ^ fp; // cross product
    }
    return mp;
}


// ── TEST: Forces ──────────────────────────────────────────────────────────────

TEST_CASE("Forces - pressure force matches OpenFOAM reference")
{
    Foam::Time& runTime = *timePtr;

    // Parametrize over all available executors (Serial, CPU, GPU)
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("uniform pressure, fixedWalls" + execName)
    {
        // Register MeshAdapter in the object registry
        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;

        // Create a uniform pressure field registered in the mesh
        Foam::volScalarField ofP(
            Foam::IOobject(
                "p",
                runTime.timeName(),
                mesh,
                Foam::IOobject::MUST_READ,
                Foam::IOobject::NO_WRITE
            ),
            mesh
        );
        // Set uniform value on internal field; BCs compute boundary values
        ofP.primitiveFieldRef() = Foam::scalar {1.0};
        ofP.correctBoundaryConditions();

        // Register the NeoN pressure field in the VectorCollection
        fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        nf::constructAndRegister(vc, rt, ofP, false);

        // Build a dictionary that matches the Forces constructor expectations
        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", Foam::scalar {1.0});
        dict.add("pRef", Foam::scalar {0.0});
        dict.add("CofR", Foam::vector(0, 0, 0));

        // Construct Forces via the same signature as the RTST path
        nf::Forces forces("neoForces", runTime, dict);

        REQUIRE(forces.execute());

        const nf::ForceResult& res = forces.lastResult();

        // Compute OpenFOAM reference
        const Foam::scalar rhoRef = 1.0;
        const Foam::scalar pRef = 0.0;
        const Foam::vector cofR = Foam::vector::zero;
        Foam::vector ofFp = ofPressureForce(ofP, mesh, "fixedWalls", rhoRef, pRef);
        Foam::vector ofMp = ofPressureMoment(ofP, mesh, "fixedWalls", rhoRef, pRef, cofR);

        const double tol = 1e-10;
        CHECK(res.pressureForce[0] == Catch::Approx(ofFp[0]).margin(tol));
        CHECK(res.pressureForce[1] == Catch::Approx(ofFp[1]).margin(tol));
        CHECK(res.pressureForce[2] == Catch::Approx(ofFp[2]).margin(tol));

        CHECK(res.pressureMoment[0] == Catch::Approx(ofMp[0]).margin(tol));
        CHECK(res.pressureMoment[1] == Catch::Approx(ofMp[1]).margin(tol));
        CHECK(res.pressureMoment[2] == Catch::Approx(ofMp[2]).margin(tol));
    }

    SECTION("random pressure field, fixedWalls" + execName)
    {
        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;

        // Create a random pressure field
        auto ofP = randomScalarField(runTime, mesh, "p");
        ofP.correctBoundaryConditions();

        // Register the NeoN pressure field in the VectorCollection
        fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        nf::constructAndRegister(vc, rt, ofP, false);

        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", Foam::scalar {1.225});
        dict.add("pRef", Foam::scalar {0.5});
        dict.add("CofR", Foam::vector(0.1, 0.2, 0.3));

        nf::Forces forces("neoForces", runTime, dict);
        REQUIRE(forces.execute());

        const nf::ForceResult& res = forces.lastResult();

        Foam::scalar rhoRef = 1.225;
        Foam::scalar pRef = 0.5;
        Foam::vector cofR(0.1, 0.2, 0.3);
        Foam::vector ofFp = ofPressureForce(ofP, mesh, "fixedWalls", rhoRef, pRef);
        Foam::vector ofMp = ofPressureMoment(ofP, mesh, "fixedWalls", rhoRef, pRef, cofR);

        const double tol = 1e-10;
        CHECK(res.pressureForce[0] == Catch::Approx(ofFp[0]).margin(tol));
        CHECK(res.pressureForce[1] == Catch::Approx(ofFp[1]).margin(tol));
        CHECK(res.pressureForce[2] == Catch::Approx(ofFp[2]).margin(tol));

        CHECK(res.pressureMoment[0] == Catch::Approx(ofMp[0]).margin(tol));
        CHECK(res.pressureMoment[1] == Catch::Approx(ofMp[1]).margin(tol));
        CHECK(res.pressureMoment[2] == Catch::Approx(ofMp[2]).margin(tol));
    }

    SECTION("multiple patches: fixedWalls + inlet + outlet" + execName)
    {
        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;

        auto ofP = randomScalarField(runTime, mesh, "p");
        ofP.correctBoundaryConditions();

        // Register the NeoN pressure field in the VectorCollection
        fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        nf::constructAndRegister(vc, rt, ofP, false);

        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls", "inlet", "outlet"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", Foam::scalar {1.0});
        dict.add("pRef", Foam::scalar {0.0});
        dict.add("CofR", Foam::vector(0, 0, 0));

        nf::Forces forces("neoForces", runTime, dict);
        REQUIRE(forces.execute());

        const nf::ForceResult& res = forces.lastResult();

        // OpenFOAM reference: sum over all three patches
        Foam::vector ofFpTotal = Foam::vector::zero;
        for (const Foam::word& pName : Foam::wordList {"fixedWalls", "inlet", "outlet"})
        {
            ofFpTotal += ofPressureForce(ofP, mesh, pName, 1.0, 0.0);
        }

        const double tol = 1e-10;
        CHECK(res.pressureForce[0] == Catch::Approx(ofFpTotal[0]).margin(tol));
        CHECK(res.pressureForce[1] == Catch::Approx(ofFpTotal[1]).margin(tol));
        CHECK(res.pressureForce[2] == Catch::Approx(ofFpTotal[2]).margin(tol));
    }
}


// ── TEST: ForceCoeffs ─────────────────────────────────────────────────────────

TEST_CASE("ForceCoeffs - normalised coefficients match manual computation")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("random pressure, coefficient normalisation" + execName)
    {
        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;

        auto ofP = randomScalarField(runTime, mesh, "p");
        ofP.correctBoundaryConditions();

        // Register the NeoN pressure field in the VectorCollection
        fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        nf::constructAndRegister(vc, rt, ofP, false);

        const Foam::scalar rhoRef = 1.225;
        const Foam::scalar magUInf = 10.0;
        const Foam::scalar lRef = 0.5;
        const Foam::scalar Aref = 0.25;
        const Foam::scalar pRef = 0.0;
        const Foam::vector cofR = Foam::vector::zero;

        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", rhoRef);
        dict.add("pRef", pRef);
        dict.add("magUInf", magUInf);
        dict.add("lRef", lRef);
        dict.add("Aref", Aref);
        dict.add("CofR", cofR);

        nf::ForceCoeffs fc("neoForceCoeffs", runTime, dict);
        REQUIRE(fc.execute());

        // Raw force from OpenFOAM reference
        Foam::vector ofFp = ofPressureForce(ofP, mesh, "fixedWalls", rhoRef, pRef);

        // NeoFOAM raw force must match
        const nf::ForceResult& raw = fc.lastResult();
        const double tol = 1e-10;
        CHECK(raw.pressureForce[0] == Catch::Approx(ofFp[0]).margin(tol));
        CHECK(raw.pressureForce[1] == Catch::Approx(ofFp[1]).margin(tol));
        CHECK(raw.pressureForce[2] == Catch::Approx(ofFp[2]).margin(tol));
    }
}
