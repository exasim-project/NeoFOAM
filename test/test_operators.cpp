// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "fv.H"
#include "gaussConvectionScheme.H"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

NeoN::TokenList interpolationScheme;

TEST_CASE("Interpolation")
{
    Foam::Time& runTime = *timePtr;
    Foam::argList& args = *argsPtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    auto nfMesh = mesh.nfMesh();

    auto ofT = randomScalarField(runTime, mesh, "T");
    auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);

    auto zero = [](auto& field, auto value)
    {
        NeoN::fill(field.internalVector(), value);
        NeoN::fill(field.boundaryData().value(), value);
    };

    // linear interpolation hardcoded for now
    Foam::IStringStream is("linear");
    SECTION("Linear SurfaceInterpolation[scalar] on " + execName)
    {
        Foam::tmp<Foam::surfaceInterpolationScheme<Foam::scalar>> foamInterPol =
            Foam::surfaceInterpolationScheme<Foam::scalar>::New(mesh, is);
        Foam::surfaceScalarField ofSurfT(foamInterPol->interpolate(ofT));
        auto nfSurfT = NeoFOAM::constructFrom(exec, nfMesh, ofSurfT);
        zero(nfSurfT, 0.0);

        interpolationScheme.insert(std::string("linear"));
        auto linearKernel = fvcc::SurfaceInterpolationFactory<NeoN::scalar>::create(
            exec,
            nfMesh,
            interpolationScheme
        );
        auto op = fvcc::SurfaceInterpolation(exec, nfMesh, std::move(linearKernel));
        op.interpolate(nfT, nfSurfT);
        nfSurfT.correctBoundaryConditions();

        NeoFOAM::compare(nfSurfT, ofSurfT, ApproxScalar(1e-15), false);
    }

    SECTION("GaussGreenGrad[scalar] on " + execName)
    {
        Foam::fv::gaussGrad<Foam::scalar> foamGradScalar(mesh, is);
        Foam::volVectorField ofGradT("ofGradT", foamGradScalar.calcGrad(ofT, "test"));

        auto nfGradT = NeoFOAM::constructFrom(exec, nfMesh, ofGradT);
        zero(nfGradT, NeoN::Vec3(0.0, 0.0, 0.0));

        fvcc::GaussGreenGrad(exec, nfMesh).grad(nfT, NeoN::dsl::Coeff(), nfGradT.internalVector());
        nfGradT.correctBoundaryConditions();
        auto nfGradTHost = nfGradT.internalVector().copyToHost();
        for (size_t celli = 0; celli < nfGradTHost.size(); celli++)
        {
            REQUIRE(nfGradTHost.view()[celli][0] == Catch::Approx(ofGradT[celli][0]).margin(1e-15));
            REQUIRE(nfGradTHost.view()[celli][1] == Catch::Approx(ofGradT[celli][1]).margin(1e-15));
            // NOTE: we relax test in z direction, OpenFOAM explicitly zeros out in 2D case
            REQUIRE(nfGradTHost.view()[celli][2] == Catch::Approx(ofGradT[celli][2]).margin(1e-6));
        }

        // NOTE not using compare for now since it has same tolerance in all directions
        // NeoFOAM::compare(nfGradT, ofGradT, ApproxVector(1e-15), false);
    }

    auto ofPhi = randDimField<Foam::surfaceScalarField>(mesh, Foam::dimless, "phi");
    auto nfPhi = NeoFOAM::constructFrom(exec, nfMesh, ofPhi);

    SECTION("GaussGreenDiv[scalar] on " + execName)
    {
        // NOTE: seems like copy construction results in hanging tests
        auto foamDivScalar = Foam::fv::gaussConvectionScheme<Foam::scalar>(mesh, ofPhi, is);
        Foam::volScalarField ofDivT("ofDivT", foamDivScalar.fvcDiv(ofPhi, ofT));
        auto nfDivT = NeoFOAM::constructFrom(exec, nfMesh, ofDivT);
        zero(nfDivT, 0.0);

        fvcc::GaussGreenDiv<NeoN::scalar>(exec, nfMesh, interpolationScheme)
            .div(nfDivT, nfPhi, nfT, dsl::Coeff(1.0));
        nfDivT.correctBoundaryConditions();

        NeoFOAM::compare(nfDivT, ofDivT, ApproxScalar(1e-15), false);
    }


    SECTION("linear GaussGreen from expression on " + execName)
    {
        auto foamDivScalar = Foam::fv::gaussConvectionScheme<Foam::scalar>(mesh, ofPhi, is);
        Foam::volScalarField ofDivT("ofDivT", foamDivScalar.fvcDiv(ofPhi, ofT));
        NeoN::TokenList scheme = NeoN::TokenList({std::string("Gauss"), std::string("linear")});

        auto nfDivT = NeoFOAM::constructFrom(exec, nfMesh, ofDivT);
        zero(nfDivT, 0.0);

        NeoN::dsl::SpatialOperator divOp = dsl::exp::div(nfPhi, nfT);
        divOp.read(scheme);
        divOp.explicitOperation(nfDivT.internalVector());
        nfDivT.correctBoundaryConditions();

        NeoFOAM::compare(nfDivT, ofDivT, ApproxScalar(1e-15), false);
    }
}
