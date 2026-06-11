// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include <algorithm>
#include <cmath>

#include "common.hpp"

#include "fv.H"
#include "snGradScheme.H"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr; // A single time object (parallel runtime set up by the MPI catch main)

// Distributed snGrad regression test. The serial test_snGrad cannot exercise processor faces; this
// partitioned variant compares NeoN's snGrad against OpenFOAM including the processor-patch values.
//
// The setup_snGrad mesh is a sheared (non-orthogonal) parallelogram, decomposed across 3 ranks via
// scotch, so the processor faces are non-orthogonal — this is the config that exercises the full
// processor-boundary geometry:
//   * uncorrected: exact owner-to-neighbour deltaCoeffs 1/|Cnei - Cown|,
//   * corrected / limitedCorrected: the non-orthogonal correction corrVec . interpolate(grad)
//     applied at processor faces via the neighbour-gradient halo.
//
// All three schemes below match OpenFOAM to ~1e-12 across processor boundaries on this sheared
// scotch decomposition.
//
// The parallel sanity check is a SEPARATE test case on purpose: it keeps the binary's exit code
// well-defined even when every section is skipped (Catch2 otherwise reports the all-skipped
// exit code 4).
TEST_CASE("Distributed snGrad parallel sanity")
{
    REQUIRE(Foam::Pstream::parRun());
    REQUIRE(Foam::Pstream::nProcs() == 3);
}

TEST_CASE("Distributed snGrad schemes")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& nfMesh = rt.nfMesh;

    auto ofU = nf::randomVectorField(runTime, mesh, "U");
    auto nfU = nf::constructFrom(rt.exec, nfMesh, ofU);

    // Verify the processor-patch halo directly: constructFrom() runs
    // nfU.correctBoundaryConditions() → setProcBoundaryValue (seeds owner) then the batched
    // processor-halo exchange, so nfU.boundaryData().value() proc-tail must equal OpenFOAM's
    // processor-patch value (the neighbour cell value), NOT the owner value. This checks the
    // processor-halo exchange independently of any snGrad scheme.
    {
        Foam::label nNonProcBndFaces = 0;
        forAll(ofU.boundaryField(), patchi)
        {
            if (ofU.boundaryField()[patchi].patch().type() != "processor")
                nNonProcBndFaces += ofU.boundaryField()[patchi].size();
        }
        auto haloH = nfU.boundaryData().value().copyToHost();
        auto haloView = haloH.view();

        std::size_t haloIdx = static_cast<std::size_t>(nNonProcBndFaces);
        double worstVsNei = 0.0;
        forAll(ofU.boundaryField(), patchi)
        {
            const auto& pin = ofU.boundaryField()[patchi];
            if (pin.patch().type() != "processor") continue;
            for (Foam::label i = 0; i < pin.size(); ++i)
            {
                const auto h = haloView[haloIdx++]; // NeoN halo
                const Foam::vector nei = pin[i];    // OF neighbour value (the true halo)
                const double dNei = std::sqrt(
                    (h[0] - nei.x()) * (h[0] - nei.x()) + (h[1] - nei.y()) * (h[1] - nei.y())
                    + (h[2] - nei.z()) * (h[2] - nei.z())
                );
                worstVsNei = std::max(worstVsNei, dNei);
            }
        }
        REQUIRE(worstVsNei < 1e-12);
    }

    auto zeroSurface = [](auto& field, auto zeroVal)
    {
        NeoN::fill(field.internalVector(), zeroVal);
        NeoN::fill(field.boundaryData().value(), zeroVal);
    };

    // Builds OF and NeoN snGrad(U) for the given scheme and asserts they agree on internal faces
    // AND on every boundary patch — non-processor first, then processor (EqualsBoundary ordering).
    auto checkScheme = [&](const std::string& ofScheme, NeoN::Input nfInput)
    {
        Foam::IStringStream is(ofScheme);
        auto tSnGradU = Foam::fv::snGradScheme<Foam::vector>::New(mesh, is)->snGrad(ofU);
        const Foam::surfaceVectorField& ofSnGradU = tSnGradU.cref();

        auto nfSnGradU = nf::constructFrom(rt.exec, nfMesh, ofSnGradU);
        zeroSurface(nfSnGradU, NeoN::Vec3 {0.0, 0.0, 0.0});

        fvcc::FaceNormalGradientFactory<NeoN::Vec3>::create(exec, nfMesh, nfInput)
            ->faceNormalGrad(nfU, nfSnGradU);

        REQUIRE_THAT(nfSnGradU, EqualsInternal(ofSnGradU, ApproxVector(1e-12)));
        REQUIRE_THAT(nfSnGradU.boundaryData(), EqualsBoundary(ofSnGradU, ApproxVector(1e-12)));
    };

    SECTION("uncorrected Vec3 matches OpenFOAM across processor boundaries on " + execName)
    {
        checkScheme("uncorrected", NeoN::TokenList({std::string("uncorrected")}));
    }

    SECTION("corrected Vec3 matches OpenFOAM across processor boundaries on " + execName)
    {
        checkScheme("corrected", NeoN::TokenList({std::string("corrected")}));
    }

    SECTION(
        "limited corrected Vec3 (0.5) matches OpenFOAM across processor boundaries on " + execName
    )
    {
        checkScheme(
            "limited corrected 0.5",
            NeoN::TokenList({std::string("limited"), std::string("corrected"), NeoN::scalar(0.5)})
        );
    }
}
