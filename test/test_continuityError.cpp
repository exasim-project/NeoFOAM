// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"
#include "NeoFOAM/auxiliary/continuityError.hpp"

namespace fvc = Foam::fvc;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

TEST_CASE("ContinuityError")
{
    Foam::Time& runTime = *timePtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);

    auto ofU = NeoFOAM::randomVectorField(runTime, mesh, "U");
    ofU.correctBoundaryConditions();

    Foam::surfaceScalarField ofPhi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(ofU)
    );
    ofPhi.correctBoundaryConditions();

    auto nfPhi = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi);

    SECTION("computeContinuityError matches OpenFOAM continuityErrs.H " + execName)
    {
        // OpenFOAM reference — replicates src/finiteVolume/cfdTools/incompressible/continuityErrs.H
        Foam::volScalarField contErr(fvc::div(ofPhi));
        const Foam::scalar ofSumLocal =
            runTime.deltaTValue() * Foam::mag(contErr)().weightedAverage(mesh.V()).value();
        const Foam::scalar ofGlobal =
            runTime.deltaTValue() * contErr.weightedAverage(mesh.V()).value();

        const auto nfErrs = nf::computeContinuityError(nfPhi, rt);

        REQUIRE(nfErrs.sumLocal == Catch::Approx(ofSumLocal).margin(1e-12));
        REQUIRE(nfErrs.global == Catch::Approx(ofGlobal).margin(1e-12));
    }

    SECTION("reportContinuityError accumulates global error " + execName)
    {
        const auto nfErrs = nf::computeContinuityError(nfPhi, rt);

        NeoN::scalar cumulative = 0.0;
        nf::reportContinuityError(nfPhi, rt, cumulative);
        REQUIRE(cumulative == Catch::Approx(nfErrs.global).margin(1e-15));

        nf::reportContinuityError(nfPhi, rt, cumulative);
        REQUIRE(cumulative == Catch::Approx(2.0 * nfErrs.global).margin(1e-15));
    }
}
