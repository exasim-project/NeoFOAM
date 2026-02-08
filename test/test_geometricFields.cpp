// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

extern Foam::Time* timePtr; // A single time object

TEST_CASE("VolumeField")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    auto nfMesh = mesh.nfMesh();

    auto ofT = randomScalarField(runTime, mesh, "T");
    auto ofU = randomVectorField(runTime, mesh, "U");
    auto ofPhi = randomSurfaceScalarField(runTime, mesh, "phi");

    SECTION("volumeScalarField " + execName)
    {
        auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);
        NeoFOAM::compare(nfT, ofT, ApproxScalar(1e-15));
    }

    SECTION("volumeVectorField " + execName)
    {
        auto nfU = NeoFOAM::constructFrom(exec, nfMesh, ofU);
        NeoFOAM::compare(nfU, ofU, ApproxVector(1e-15));
    }
    SECTION("surfaceScalarField " + execName)
    {
        auto nfPhi = NeoFOAM::constructFrom(exec, nfMesh, ofPhi);
        NeoFOAM::compare(nfPhi, ofPhi, ApproxScalar(1e-15));
    }
}
