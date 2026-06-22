// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Compares NeoFOAM's KqRWallFunction patch BC against OpenFOAM's
// kqRWallFunctionFvPatchField<scalar>. Both are nominally zero-gradient — this
// test pins that behaviour by reading the same `k` field through both stacks
// and asserting the boundary values match after correctBoundaryConditions().

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "fvCFD.H"

using Catch::Approx;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

TEST_CASE("kqRWallFunction: NeoFOAM boundary correction matches OpenFOAM")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    auto nfMesh = mesh.nfMesh();

    // Read the OF k field. The 0/k file in setup_operator declares
    // `type kqRWallFunction` on `fixedWalls`, so OpenFOAM's runtime selection
    // (libturbulenceModels) constructs a kqRWallFunctionFvPatchField<scalar>
    // for that patch as the field is read.
    Foam::volScalarField ofK(
        Foam::IOobject(
            "k",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh
    );

    // Locate the wall patch we expect to carry the kqRWallFunction BC.
    const Foam::label wallId = mesh.boundaryMesh().findPatchID("fixedWalls");
    REQUIRE(wallId >= 0);
    REQUIRE(ofK.boundaryField()[wallId].type() == "kqRWallFunction");

    // Evaluate the BC on the OF side: zero-gradient pulls each boundary face's
    // value from its adjacent internal cell.
    ofK.correctBoundaryConditions();

    SECTION("boundary values agree after correction (" + execName + ")")
    {
        // Round-trip the field through the NeoFOAM reader. This is what
        // `auxiliary/readers.hpp` does in the production code path: the
        // OF dictionary entry `type kqRWallFunction` is converted by the
        // patchInserter into a NeoN dict `{type: kqRWallFunction}`, which
        // the VolumeBoundaryFactory dispatches to KqRWallFunction.
        auto nfK = NeoFOAM::constructFrom(exec, nfMesh, ofK);

        // Sanity: the internal field round-tripped exactly.
        REQUIRE_THAT(nfK, EqualsInternal(ofK, ApproxScalar(1e-15)));

        // Re-apply the NF boundary correction so the comparison reflects
        // KqRWallFunction::correctBoundaryCondition(), not whatever
        // constructFrom() copied at construction time.
        nfK.correctBoundaryConditions();

        REQUIRE_THAT(nfK.boundaryData(), EqualsBoundary(ofK, ApproxScalar(1e-14)));
    }

    SECTION("KqRWallFunction is registered with the volume factory")
    {
        // If readers.hpp emits `type kqRWallFunction` but no derived class is
        // registered under that name, VolumeBoundaryFactory::create() throws
        // at constructFrom() time. Guard against silent regressions.
        const auto& table = fvcc::VolumeBoundaryFactory<NeoN::scalar>::table();
        REQUIRE(table.find("kqRWallFunction") != table.end());
    }
}
