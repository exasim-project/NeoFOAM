// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "NeoFOAM/auxiliary/bound.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

// bound() ports OpenFOAM's Foam::bound(): cells that undershoot to zero or below are refilled
// from the neighbourhood rather than pinned at the floor. The distinction matters because the
// turbulence models divide by these fields — a cell pinned at a floor orders of magnitude below
// the physical scale makes nu_t explode.
TEST_CASE("bound")
{
    Foam::Time& runTime = *timePtr;

    auto exec = NeoN::SerialExecutor {};
    auto meshPtrLocal = NeoFOAM::createMesh(exec, runTime);
    NeoN::UnstructuredMesh& mesh = meshPtrLocal->nfMesh();

    const auto nCells = mesh.nCells();
    REQUIRE(nCells > 2);

    auto makeField = [&](const std::vector<NeoN::scalar>& values)
    {
        auto field = fvcc::VolumeField<NeoN::scalar>(
            exec,
            "phi",
            mesh,
            fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh)
        );
        field.internalVector() = NeoN::Vector<NeoN::scalar>(exec, values);
        NeoN::fill(field.boundaryData().value(), NeoN::scalar(1));
        return field;
    };

    SECTION("a field already within bounds is left untouched")
    {
        std::vector<NeoN::scalar> values(static_cast<std::size_t>(nCells), 1.0);
        auto field = makeField(values);

        REQUIRE(NeoFOAM::bound(field, 0.0) == false);

        auto host = field.internalVector().copyToHost();
        for (NeoN::localIdx i = 0; i < nCells; i++)
        {
            REQUIRE(host.view()[i] == Catch::Approx(1.0).margin(1e-12));
        }
    }

    SECTION("a negative cell is refilled from the neighbourhood, not pinned at the floor")
    {
        std::vector<NeoN::scalar> values(static_cast<std::size_t>(nCells), 1.0);
        values[1] = -5.0;
        auto field = makeField(values);

        REQUIRE(NeoFOAM::bound(field, 0.0) == true);

        auto host = field.internalVector().copyToHost();
        const NeoN::scalar repaired = host.view()[1];

        // fvc::average interpolates the FLOORED field, so the cell's own floored value (0) is
        // blended with its neighbours' 1 across each face: the refill lands between 0.5 and 1,
        // not at 1. What matters is that it is a neighbourhood-scale value — a plain clamp would
        // have left this cell sitting at the lower bound.
        REQUIRE(repaired >= 0.5);
        REQUIRE(repaired <= 1.0);

        // Every other cell keeps its value.
        for (NeoN::localIdx i = 0; i < nCells; i++)
        {
            if (i == 1) continue;
            REQUIRE(host.view()[i] == Catch::Approx(1.0).margin(1e-12));
        }
    }

    SECTION("a positive cell below the bound is floored, matching OpenFOAM's pos0 selection")
    {
        std::vector<NeoN::scalar> values(static_cast<std::size_t>(nCells), 1.0);
        values[1] = 1e-12; // positive but below the bound -> floored, NOT refilled
        values[2] = -1.0;  // non-positive -> refilled from the neighbourhood
        auto field = makeField(values);

        REQUIRE(NeoFOAM::bound(field, 1e-3) == true);

        auto host = field.internalVector().copyToHost();
        REQUIRE(host.view()[1] == Catch::Approx(1e-3).margin(1e-15));
        REQUIRE(host.view()[2] > 1e-3);
    }
}
