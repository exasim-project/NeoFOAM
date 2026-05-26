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

    // Helper: zero-fill a NeoFOAM surface field with the supplied zero value
    auto zeroSurface = [](auto& field, auto zeroVal)
    {
        NeoN::fill(field.internalVector(), zeroVal);
        NeoN::fill(field.boundaryData().value(), zeroVal);
    };

    SECTION("uncorrected matches OpenFOAM on " + execName)
    {
        // OpenFOAM: explicit uncorrected snGrad
        Foam::IStringStream is("uncorrected");
        auto tSnGradT = Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);
        const Foam::surfaceScalarField& ofSnGradT = tSnGradT.cref();

        // NeoFOAM: uncorrected scheme
        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT, NeoN::scalar(0.0));

        NeoN::Input input = NeoN::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        // On the orthogonal setup_operator mesh: nonOrthDeltaCoeffs == deltaCoeffs
        NeoFOAM::compare(nfSnGradT, ofSnGradT, ApproxScalar(1e-15), true);
    }

    SECTION("corrected matches OpenFOAM on " + execName)
    {
        // OpenFOAM: explicit corrected snGrad
        Foam::IStringStream is("corrected");
        auto tSnGradT = Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);
        const Foam::surfaceScalarField& ofSnGradT = tSnGradT.cref();

        // NeoFOAM: corrected scheme
        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT, NeoN::scalar(0.0));

        NeoN::Input input = NeoN::TokenList({std::string("corrected")});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        // On orthogonal mesh corrVec = 0, so corrected == uncorrected; gradient
        // computation is exact when correction vanishes
        NeoFOAM::compare(nfSnGradT, ofSnGradT, ApproxScalar(1e-12), true);
    }

    SECTION("limited (0.333) matches OpenFOAM limited on " + execName)
    {
        // Terse OF form: "limited <coeff>"
        Foam::IStringStream is("limited 0.333");
        auto tSnGradT = Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);
        const Foam::surfaceScalarField& ofSnGradT = tSnGradT.cref();

        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT, NeoN::scalar(0.0));

        NeoN::Input input = NeoN::TokenList({std::string("limited"), NeoN::scalar(0.333)});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        NeoFOAM::compare(nfSnGradT, ofSnGradT, ApproxScalar(1e-12), true);
    }

    SECTION("limited corrected (0.5) — verbose OF form matches on " + execName)
    {
        // Verbose OF form: "limited corrected <coeff>" — the form that appears inside
        // laplacianSchemes entries like "Gauss linear limited corrected 0.5"
        Foam::IStringStream is("limited corrected 0.5");
        auto tSnGradT = Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);
        const Foam::surfaceScalarField& ofSnGradT = tSnGradT.cref();

        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT, NeoN::scalar(0.0));

        NeoN::Input input =
            NeoN::TokenList({std::string("limited"), std::string("corrected"), NeoN::scalar(0.5)});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        NeoFOAM::compare(nfSnGradT, ofSnGradT, ApproxScalar(1e-12), true);
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

        NeoFOAM::compare(nfSnGradU, ofSnGradU, ApproxVector(1e-15), true);
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

        // Orthogonal mesh: corrVec = 0, so corrected == uncorrected for Vec3 too
        NeoFOAM::compare(nfSnGradU, ofSnGradU, ApproxVector(1e-12), true);
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

        NeoFOAM::compare(nfSnGradU, ofSnGradU, ApproxVector(1e-12), true);
    }
}
