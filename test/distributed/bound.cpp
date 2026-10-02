// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "NeoFOAM/auxiliary/bound.hpp"
#include "bound.H"

namespace nf = NeoFOAM;

using NeoFOAM::EqualsInternal;

extern Foam::Time* timePtr; // A single time object (parallel runtime set up by the MPI catch main)

// Distributed bound() regression test. The serial test_bound cannot exercise processor faces: the
// refill value is fvc::average(max(vsf, lowerBound)), so a cell touching a processor boundary
// draws part of its face-area-weighted average from the other rank. The proc tail of
// boundaryData().value() holds the neighbour GHOST-CELL value, not a face value, so it has to be
// interpolated with the boundary weight before entering the average; treating it as a face value
// makes the repaired cell depend on where the decomposition happens to cut.
//
// Only the processor-face cells ON RANK 0 are driven negative, over an otherwise smooth, strictly
// positive field. That asymmetry is what makes the test discriminating: those cells are the ones
// bound() refills, and the ghost value they pull across the processor face is a large positive
// neighbour value, so the interpolated face value w*lowerBound + (1-w)*ghost differs from the raw
// ghost. Drive BOTH sides negative instead and every proc face is floored to the same
// lowerBound — the two variants then agree and the test proves nothing.
//
// OpenFOAM's Foam::bound() on the same decomposed field is the reference.
//
// The parallel sanity check is a SEPARATE test case on purpose: it keeps the binary's exit code
// well-defined even when every section is skipped (Catch2 otherwise reports the all-skipped
// exit code 4).
TEST_CASE("Distributed bound parallel sanity")
{
    REQUIRE(Foam::Pstream::parRun());
    REQUIRE(Foam::Pstream::nProcs() == 3);
}

TEST_CASE("Distributed bound matches OpenFOAM across processor faces")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;

    // Read p for its boundary types, then overwrite every value: a smooth, strictly positive
    // function of position, so owner and ghost differ across every processor face.
    auto ofPhi = randomScalarField(runTime, mesh, "p");
    forAll(ofPhi, celli)
    {
        const auto& c = mesh.C()[celli];
        ofPhi[celli] = 1.0 + c.x() + 2.0 * c.y() + 3.0 * c.z();
    }

    Foam::label nRefilled = 0;
    if (Foam::Pstream::myProcNo() == 0)
    {
        forAll(mesh.boundary(), patchi)
        {
            const auto& patch = mesh.boundary()[patchi];
            if (patch.type() != "processor") continue;
            forAll(patch.faceCells(), i)
            {
                ofPhi[patch.faceCells()[i]] = -1.0;
                nRefilled++;
            }
        }
        REQUIRE(nRefilled > 0);
    }
    ofPhi.correctBoundaryConditions();

    const NeoN::scalar lowerBound = 1e-3;

    // NeoN
    auto nfPhi = nf::constructFrom(rt.exec, rt.nfMesh, ofPhi);
    REQUIRE(nf::bound(nfPhi, lowerBound) == true);

    // OpenFOAM reference on the same decomposed field
    Foam::volScalarField ofRef(ofPhi);
    Foam::bound(ofRef, Foam::dimensionedScalar("lb", ofPhi.dimensions(), lowerBound));

    REQUIRE_THAT(nfPhi, EqualsInternal(ofRef, ApproxScalar(1e-12)));
}
