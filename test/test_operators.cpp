// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "fv.H"
#include "gaussConvectionScheme.H"
#include "gaussLaplacianScheme.H"

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
    auto ofU = randomVectorField(runTime, mesh, "U");
    auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);
    auto ofGamma = randomScalarField(runTime, mesh, "Gamma");
    auto nfGamma = NeoFOAM::constructFrom(exec, nfMesh, ofGamma);

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
        auto nfSurfT = NeoFOAM::constructSurfaceField(exec, nfMesh, ofSurfT);
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

    Foam::surfaceScalarField ofPhi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("phi", Foam::dimless, 0.0)
    );
    auto nfPhi = NeoFOAM::constructSurfaceField(exec, nfMesh, ofPhi);

    SECTION("GaussGreenDiv[scalar] on " + execName)
    {
        // NOTE: seems like copy construction results in hanging tests
        Foam::fv::gaussConvectionScheme<Foam::scalar> foamDivScalar(mesh, ofPhi, is);
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
        Foam::fv::gaussConvectionScheme<Foam::scalar> foamDivScalar(mesh, ofPhi, is);
        Foam::volScalarField ofDivT("ofDivT", foamDivScalar.fvcDiv(ofPhi, ofT));
        NeoN::TokenList scheme = NeoN::TokenList({std::string("Gauss"), std::string("linear")});

        auto nfDivT = NeoFOAM::constructFrom(exec, nfMesh, ofDivT);
        zero(nfDivT, 0.0);

        NeoN::dsl::SpatialOperator divOp = dsl::exp::div(nfPhi, nfT);
        divOp.read(scheme);
        divOp.explicitOperation(nfDivT.internalVector());
        nfDivT.correctBoundaryConditions();

        NeoFOAM::compare(nfDivT, ofDivT, ApproxScalar(1e-15), false);
 //   }
 //   SECTION("GaussGreenLap[scalar] on " + execName)
 //   {
        Foam::IStringStream is2("linear");
	Foam::tmp<Foam::surfaceInterpolationScheme<Foam::scalar>> foamInterPol2 =
            Foam::surfaceInterpolationScheme<Foam::scalar>::New(mesh, is2);
        Foam::IStringStream lapData("linear uncorrected");
        Foam::fv::gaussLaplacianScheme<Foam::scalar, Foam::scalar> foamLapScalar(mesh, lapData);
        Foam::surfaceScalarField ofSurfGamma(foamInterPol2->interpolate(ofGamma));
        Foam::volScalarField ofLapT("ofLapT", foamLapScalar.fvcLaplacian(ofSurfGamma, ofT));

        auto nfLapT = NeoFOAM::constructFrom(exec, nfMesh, ofLapT);
        zero(nfLapT, 0.0);
        auto nfSurfGamma = NeoFOAM::constructSurfaceField(exec, nfMesh, ofSurfGamma);

	NeoFOAM::compare(nfSurfGamma, ofSurfGamma, ApproxScalar(1e-15), false);

	NeoN::TokenList lapScheme = NeoN::TokenList({std::string("linear"), std::string("uncorrected")});
        fvcc::GaussGreenLaplacian<NeoN::scalar>(exec, nfMesh, lapScheme)
            .laplacian(nfLapT, nfSurfGamma, nfT, dsl::Coeff());
        nfLapT.correctBoundaryConditions();

        NeoFOAM::compare(nfLapT, ofLapT, ApproxScalar(1e-15), false);   
////////////////////////
	Foam::volScalarField ofConvDiffT(
            "ofConvDiffT",
            ofDivT + ofLapT
        );
	auto nfConvDiffSplit =
            NeoFOAM::constructFrom(exec, nfMesh, ofConvDiffT);
	zero(nfConvDiffSplit, 0.0);
	// element-wise: nfConvDiffSplit = nfDivT + nfLapT
        {
            auto execN = nfConvDiffSplit.exec();
            auto [out, a, b] = views(
            nfConvDiffSplit.internalVector(),
            nfDivT.internalVector(),
            nfLapT.internalVector()
        );

        NeoN::parallelFor(
            execN,
            {0, out.size()},
            KOKKOS_LAMBDA(const size_t i) {
                out[i] = a[i] + b[i];
            }
        );
        }

        // apply BCs
        nfConvDiffSplit.correctBoundaryConditions();
//	auto nfConvDiffSplit = nfDivT + nfLapT;
        auto nfConvDiffFused = nfConvDiffSplit;
	zero(nfConvDiffFused, 0.0);                   
	//NeoN::TokenList lapSchemes = NeoN::TokenList({std::string("linear"), std::string("uncorrected")});
	fvcc::GaussGreenConvDiff<NeoN::scalar> convDiffOp(exec, nfMesh, lapScheme);
        convDiffOp.convDiff(nfConvDiffFused, nfPhi, nfSurfGamma, nfT, dsl::Coeff(1.0));
        nfConvDiffFused.correctBoundaryConditions();
	NeoFOAM::compare(nfConvDiffSplit, ofConvDiffT, ApproxScalar(1e-15), false);

        // NeoN fused vs NeoN split
        //NeoFOAM::compare(nfConvDiffFused, nfConvDiffSplit, ApproxScalar(1e-15), false);

        // NeoN fused vs OpenFOAM split
        //NeoFOAM::compare(nfConvDiffFused, ofConvDiffT, ApproxScalar(1e-15), false);
    }

    //SECTION("GaussGreenConvDiff[scalar] == Div + Laplacian on " + execName)

}
