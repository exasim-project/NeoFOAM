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

    // Helper: zero-fill NeoFOAM surface field
    auto zeroSurface = [](auto& field)
    {
        NeoN::fill(field.internalVector(), NeoN::scalar(0.0));
    };

    SECTION("uncorrected matches OpenFOAM on " + execName)
    {
        // OpenFOAM: explicit uncorrected snGrad
        Foam::IStringStream is("uncorrected");
        Foam::surfaceScalarField ofSnGradT =
            Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);

        // NeoFOAM: uncorrected scheme
        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT);

        NeoN::Input input = NeoN::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        // On the orthogonal setup_operator mesh: nonOrthDeltaCoeffs == deltaCoeffs
        NeoFOAM::compare(nfSnGradT, ofSnGradT, ApproxScalar(1e-15), false);
    }

    SECTION("corrected matches OpenFOAM on " + execName)
    {
        // OpenFOAM: explicit corrected snGrad
        Foam::IStringStream is("corrected");
        Foam::surfaceScalarField ofSnGradT =
            Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);

        // NeoFOAM: corrected scheme
        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT);

        NeoN::Input input = NeoN::TokenList({std::string("corrected")});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        // On orthogonal mesh corrVec = 0, so corrected == uncorrected; gradient
        // computation is exact when correction vanishes
        NeoFOAM::compare(nfSnGradT, ofSnGradT, ApproxScalar(1e-12), false);
    }

    SECTION("limitedCorrected (0.333) matches OpenFOAM limited on " + execName)
    {
        // OpenFOAM: "limited <coeff>" — backwards-compat: coeff is a plain number
        Foam::IStringStream is("limited 0.333");
        Foam::surfaceScalarField ofSnGradT =
            Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is)->snGrad(ofT);

        // NeoFOAM: limitedCorrected scheme with same coefficient
        auto nfSnGradT = NeoFOAM::constructFrom(exec, nfMesh, ofSnGradT);
        zeroSurface(nfSnGradT);

        NeoN::Input input =
            NeoN::TokenList({std::string("limitedCorrected"), NeoN::scalar(0.333)});
        fvcc::FaceNormalGradientFactory<NeoN::scalar>::create(exec, nfMesh, input)
            ->faceNormalGrad(nfT, nfSnGradT);

        // On orthogonal mesh corrVec = 0 so limiter doesn't apply;
        // result equals uncorrected
        NeoFOAM::compare(nfSnGradT, ofSnGradT, ApproxScalar(1e-12), false);
    }
}
