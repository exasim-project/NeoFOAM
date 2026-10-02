// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "fv.H"
#include "snGradScheme.H"

namespace fvcc = NeoN::finiteVolume::cellCentred;

extern Foam::Time* timePtr;
extern Foam::argList* argsPtr;
extern Foam::fvMesh* meshPtr;

TEST_CASE("snGrad schemes")
{
    Foam::Time& runTime = *timePtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    auto nfMesh = mesh.nfMesh();

    auto ofT = randomScalarField(runTime, mesh, "T");
    auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);

    auto ofU = randomVectorField(runTime, mesh, "U");
    auto nfU = NeoFOAM::constructFrom(exec, nfMesh, ofU);

    // Zero-fill a NeoFOAM surface field
    auto zeroSurface = [](auto& field, auto zeroVal)
    {
        NeoN::fill(field.internalVector(), zeroVal);
        NeoN::fill(field.boundaryData().value(), zeroVal);
    };

    SECTION("uncorrected matches OpenFOAM on " + execName)
    {
        Foam::IStringStream is("uncorrected");
        auto tSnGradT = Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);
        const Foam::surfaceScalarField& ofSnGradT = tSnGradT.cref();

        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT, NeoN::scalar(0.0));

        NeoN::Input input = NeoN::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        // Internal faces — iterate the surface field directly as a range
        const auto nfInternal = nfSnGradT.internalVector().copyToHost();
        REQUIRE_THAT(
            nfInternal.view({0, ofSnGradT.size()}),
            Catch::Matchers::RangeEquals(ofSnGradT, ApproxScalar(1e-15))
        );

        // Boundary patches
        const auto nfBoundary = nfSnGradT.boundaryData().value().copyToHost();
        size_t patchStart = 0;
        for (const auto& patch : ofSnGradT.boundaryField())
        {
            REQUIRE_THAT(
                nfBoundary.view({patchStart, patchStart + patch.size()}),
                Catch::Matchers::RangeEquals(patch, ApproxScalar(1e-15))
            );
            patchStart += patch.size();
        }
    }

    SECTION("corrected matches OpenFOAM on " + execName)
    {
        Foam::IStringStream is("corrected");
        auto tSnGradT = Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);
        const Foam::surfaceScalarField& ofSnGradT = tSnGradT.cref();

        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT, NeoN::scalar(0.0));

        NeoN::Input input = NeoN::TokenList({std::string("corrected")});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        const auto nfInternal = nfSnGradT.internalVector().copyToHost();
        REQUIRE_THAT(
            nfInternal.view({0, ofSnGradT.size()}),
            Catch::Matchers::RangeEquals(ofSnGradT, ApproxScalar(1e-12))
        );

        const auto nfBoundary = nfSnGradT.boundaryData().value().copyToHost();
        size_t patchStart = 0;
        for (const auto& patch : ofSnGradT.boundaryField())
        {
            REQUIRE_THAT(
                nfBoundary.view({patchStart, patchStart + patch.size()}),
                Catch::Matchers::RangeEquals(patch, ApproxScalar(1e-12))
            );
            patchStart += patch.size();
        }
    }

    SECTION("limited (0.333) matches OpenFOAM limited on " + execName)
    {
        Foam::IStringStream is("limited 0.333");
        auto tSnGradT = Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);
        const Foam::surfaceScalarField& ofSnGradT = tSnGradT.cref();

        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT, NeoN::scalar(0.0));

        NeoN::Input input = NeoN::TokenList({std::string("limited"), NeoN::scalar(0.333)});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        const auto nfInternal = nfSnGradT.internalVector().copyToHost();
        REQUIRE_THAT(
            nfInternal.view({0, ofSnGradT.size()}),
            Catch::Matchers::RangeEquals(ofSnGradT, ApproxScalar(1e-12))
        );

        const auto nfBoundary = nfSnGradT.boundaryData().value().copyToHost();
        size_t patchStart = 0;
        for (const auto& patch : ofSnGradT.boundaryField())
        {
            REQUIRE_THAT(
                nfBoundary.view({patchStart, patchStart + patch.size()}),
                Catch::Matchers::RangeEquals(patch, ApproxScalar(1e-12))
            );
            patchStart += patch.size();
        }
    }

    SECTION("limited corrected (0.5) — verbose OF form matches on " + execName)
    {
        // "limited corrected <coeff>" — form used inside laplacianSchemes entries
        // like "Gauss linear limited corrected 0.5"
        Foam::IStringStream is("limited corrected 0.5");
        auto tSnGradT = Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);
        const Foam::surfaceScalarField& ofSnGradT = tSnGradT.cref();

        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT, NeoN::scalar(0.0));

        NeoN::Input input =
            NeoN::TokenList({std::string("limited"), std::string("corrected"), NeoN::scalar(0.5)});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        const auto nfInternal = nfSnGradT.internalVector().copyToHost();
        REQUIRE_THAT(
            nfInternal.view({0, ofSnGradT.size()}),
            Catch::Matchers::RangeEquals(ofSnGradT, ApproxScalar(1e-12))
        );

        const auto nfBoundary = nfSnGradT.boundaryData().value().copyToHost();
        size_t patchStart = 0;
        for (const auto& patch : ofSnGradT.boundaryField())
        {
            REQUIRE_THAT(
                nfBoundary.view({patchStart, patchStart + patch.size()}),
                Catch::Matchers::RangeEquals(patch, ApproxScalar(1e-12))
            );
            patchStart += patch.size();
        }
    }

    SECTION("uncorrected Vec3 matches OpenFOAM on " + execName)
    {
        Foam::IStringStream is("uncorrected");
        auto tSnGradU = Foam::fv::snGradScheme<Foam::vector>::New(mesh, is)->snGrad(ofU);
        const Foam::surfaceVectorField& ofSnGradU = tSnGradU.cref();

        auto nfSnGradU = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradU);
        zeroSurface(nfSnGradU, NeoN::Vec3 {0.0, 0.0, 0.0});

        NeoN::Input input = NeoN::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradientFactory<NeoN::Vec3>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfU, nfSnGradU);

        const auto nfInternal = nfSnGradU.internalVector().copyToHost();
        REQUIRE_THAT(
            nfInternal.view({0, ofSnGradU.size()}),
            Catch::Matchers::RangeEquals(ofSnGradU, ApproxVector(1e-15))
        );

        const auto nfBoundary = nfSnGradU.boundaryData().value().copyToHost();
        size_t patchStart = 0;
        for (const auto& patch : ofSnGradU.boundaryField())
        {
            REQUIRE_THAT(
                nfBoundary.view({patchStart, patchStart + patch.size()}),
                Catch::Matchers::RangeEquals(patch, ApproxVector(1e-15))
            );
            patchStart += patch.size();
        }
    }

    SECTION("corrected Vec3 matches OpenFOAM on " + execName)
    {
        Foam::IStringStream is("corrected");
        auto tSnGradU = Foam::fv::snGradScheme<Foam::vector>::New(mesh, is)->snGrad(ofU);
        const Foam::surfaceVectorField& ofSnGradU = tSnGradU.cref();

        auto nfSnGradU = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradU);
        zeroSurface(nfSnGradU, NeoN::Vec3 {0.0, 0.0, 0.0});

        NeoN::Input input = NeoN::TokenList({std::string("corrected")});
        fvcc::FaceNormalGradientFactory<NeoN::Vec3>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfU, nfSnGradU);

        const auto nfInternal = nfSnGradU.internalVector().copyToHost();
        REQUIRE_THAT(
            nfInternal.view({0, ofSnGradU.size()}),
            Catch::Matchers::RangeEquals(ofSnGradU, ApproxVector(1e-12))
        );

        const auto nfBoundary = nfSnGradU.boundaryData().value().copyToHost();
        size_t patchStart = 0;
        for (const auto& patch : ofSnGradU.boundaryField())
        {
            REQUIRE_THAT(
                nfBoundary.view({patchStart, patchStart + patch.size()}),
                Catch::Matchers::RangeEquals(patch, ApproxVector(1e-12))
            );
            patchStart += patch.size();
        }
    }

    SECTION("limited Vec3 (0.333) matches OpenFOAM limited on " + execName)
    {
        Foam::IStringStream is("limited 0.333");
        auto tSnGradU = Foam::fv::snGradScheme<Foam::vector>::New(mesh, is)->snGrad(ofU);
        const Foam::surfaceVectorField& ofSnGradU = tSnGradU.cref();

        auto nfSnGradU = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradU);
        zeroSurface(nfSnGradU, NeoN::Vec3 {0.0, 0.0, 0.0});

        NeoN::Input input = NeoN::TokenList({std::string("limited"), NeoN::scalar(0.333)});
        fvcc::FaceNormalGradientFactory<NeoN::Vec3>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfU, nfSnGradU);

        const auto nfInternal = nfSnGradU.internalVector().copyToHost();
        REQUIRE_THAT(
            nfInternal.view({0, ofSnGradU.size()}),
            Catch::Matchers::RangeEquals(ofSnGradU, ApproxVector(1e-12))
        );

        const auto nfBoundary = nfSnGradU.boundaryData().value().copyToHost();
        size_t patchStart = 0;
        for (const auto& patch : ofSnGradU.boundaryField())
        {
            REQUIRE_THAT(
                nfBoundary.view({patchStart, patchStart + patch.size()}),
                Catch::Matchers::RangeEquals(patch, ApproxVector(1e-12))
            );
            patchStart += patch.size();
        }
    }
}
