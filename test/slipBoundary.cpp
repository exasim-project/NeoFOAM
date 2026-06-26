// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Pins the `slip` boundary mapping added to auxiliary/readers.hpp. Before the
// mapping existed, reading an OpenFOAM field with `type slip;` aborted in
// readVolBoundaryConditions() with a FatalError ("Unsupported boundary
// condition type"). This exercises the production reader path
// (constructFrom -> readVolBoundaryConditions -> VolumeBoundaryFactory) for
// both scalar and vector fields and checks the corrected boundary values match
// OpenFOAM's slipFvPatchField. The DrivAre case needs slip on its ground patch
// for U/p/k/omega/nut, so this guards that the whole case fails to construct.

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "fvCFD.H"

using Catch::Approx;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

TEST_CASE("slip boundary: reader maps `slip` and matches OpenFOAM")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    auto nfMesh = mesh.nfMesh();

    const Foam::label wallId = mesh.boundaryMesh().findPatchID("fixedWalls");
    REQUIRE(wallId >= 0);

    // Build the patch-field-type list in memory so the test does not depend on
    // a slip entry living in any on-disk setup field. slip on the wall patch,
    // empty preserved where required, zeroGradient elsewhere.
    Foam::wordList patchTypes(mesh.boundary().size(), Foam::word("zeroGradient"));
    forAll(mesh.boundary(), patchi)
    {
        if (mesh.boundary()[patchi].type() == "empty")
        {
            patchTypes[patchi] = "empty";
        }
    }
    patchTypes[wallId] = "slip";

    SECTION("scalar slip round-trips and matches OpenFOAM (" + execName + ")")
    {
        // scalar slip is zero-gradient: with a uniform internal field the
        // corrected boundary value equals the internal value everywhere.
        Foam::volScalarField ofS(
            Foam::IOobject(
                "slipScalar",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("s", Foam::dimless, 3.0),
            patchTypes
        );
        REQUIRE(ofS.boundaryField()[wallId].type() == "slip");
        ofS.correctBoundaryConditions();

        // constructFrom drives readVolBoundaryConditions; this is the call that
        // FatalError-aborted before the slip mapping was added.
        auto nfS = NeoFOAM::constructFrom(exec, nfMesh, ofS);
        REQUIRE_THAT(nfS, EqualsInternal(ofS, ApproxScalar(1e-15)));

        nfS.correctBoundaryConditions();
        REQUIRE_THAT(nfS.boundaryData(), EqualsBoundary(ofS, ApproxScalar(1e-12)));
    }

    SECTION("vector slip round-trips and matches OpenFOAM (" + execName + ")")
    {
        // vector slip removes the wall-normal component (tangential projection).
        Foam::volVectorField ofV(
            Foam::IOobject(
                "slipVector",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            mesh,
            Foam::dimensionedVector("v", Foam::dimVelocity, Foam::vector(1.0, 2.0, 3.0)),
            patchTypes
        );
        REQUIRE(ofV.boundaryField()[wallId].type() == "slip");
        ofV.correctBoundaryConditions();

        auto nfV = NeoFOAM::constructFrom(exec, nfMesh, ofV);
        REQUIRE_THAT(nfV, EqualsInternal(ofV, ApproxVector(1e-14)));

        nfV.correctBoundaryConditions();
        REQUIRE_THAT(nfV.boundaryData(), EqualsBoundary(ofV, ApproxVector(1e-10)));
    }

    SECTION("slip is registered with the volume factories")
    {
        // readers.hpp emits `type slip`; if no derived class is registered under
        // that name VolumeBoundaryFactory::create() throws at constructFrom().
        const auto& sTable = fvcc::VolumeBoundaryFactory<NeoN::scalar>::table();
        REQUIRE(sTable.find("slip") != sTable.end());
        const auto& vTable = fvcc::VolumeBoundaryFactory<NeoN::Vec3>::table();
        REQUIRE(vTable.find("slip") != vTable.end());
    }
}
