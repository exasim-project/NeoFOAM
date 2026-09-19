// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "fv.H"
#include "gaussConvectionScheme.H"
#include "boundedConvectionScheme.H"

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

        REQUIRE_THAT(nfSurfT, EqualsInternal(ofSurfT, ApproxScalar(1e-15)));
    }

    SECTION("GaussGreenGrad[scalar] on " + execName)
    {
        Foam::fv::gaussGrad<Foam::scalar> foamGradScalar(mesh, is);
        Foam::volVectorField ofGradT("ofGradT", foamGradScalar.calcGrad(ofT, "test"));

        auto nfGradT = NeoFOAM::constructFrom(exec, nfMesh, ofGradT);
        zero(nfGradT, NeoN::Vec3(0.0, 0.0, 0.0));

        fvcc::GaussGreenGrad(exec, nfMesh).grad(nfT, NeoN::dsl::Coeff(), nfGradT.internalVector());
        nfGradT.correctBoundaryConditions();

        REQUIRE_THAT(nfGradT, EqualsInternal(ofGradT, ApproxVector({1e-12, 1e-12, 1e-4})));
    }

    auto ofPhi = NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, Foam::dimless, "phi");
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

        REQUIRE_THAT(nfDivT, EqualsInternal(ofDivT, ApproxScalar(1e-15)));
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

        REQUIRE_THAT(nfDivT, EqualsInternal(ofDivT, ApproxScalar(1e-15)));
    }

    // linearUpwind = upwind value + (Cf - C_upwind) & grad(phi)_upwind. The explicit divergence
    // (fvcDiv) must reproduce OpenFOAM's gaussConvectionScheme with the linearUpwind interpolation.
    // The mesh is effectively 2D (single empty z-layer), so internal-face d-vectors carry no
    // z-component and the correction is driven by the in-plane gradient that matches OF tightly.
    SECTION("GaussGreenDiv linearUpwind[scalar] on " + execName)
    {
        Foam::IStringStream isLinUpw("linearUpwind grad(T)");
        auto foamDivScalar = Foam::fv::gaussConvectionScheme<Foam::scalar>(mesh, ofPhi, isLinUpw);
        Foam::volScalarField ofDivT("ofDivT", foamDivScalar.fvcDiv(ofPhi, ofT));

        auto nfDivT = NeoFOAM::constructFrom(exec, nfMesh, ofDivT);
        zero(nfDivT, 0.0);

        NeoN::TokenList scheme =
            NeoN::TokenList({std::string("linearUpwind"), std::string("grad(T)")});
        fvcc::GaussGreenDiv<NeoN::scalar>(exec, nfMesh, scheme)
            .div(nfDivT, nfPhi, nfT, dsl::Coeff(1.0));
        nfDivT.correctBoundaryConditions();

        REQUIRE_THAT(nfDivT, EqualsInternal(ofDivT, ApproxScalar(1e-11)));
    }

    // bounded Gauss upwind = upwind divergence with the -surfaceIntegrate(phi)*T continuity-error
    // correction subtracted (Foam::fv::boundedConvectionScheme). ofPhi is a random, non-solenoidal
    // flux (not derived from a potential), so div(phi) != 0 generically and the correction is
    // actually exercised rather than vanishing.
    SECTION("GaussGreenDiv bounded[scalar] on " + execName)
    {
        Foam::IStringStream isInner("Gauss upwind");
        Foam::fv::boundedConvectionScheme<Foam::scalar> foamBoundedDivScalar(mesh, ofPhi, isInner);
        Foam::volScalarField ofDivT("ofDivT", foamBoundedDivScalar.fvcDiv(ofPhi, ofT));

        auto nfDivT = NeoFOAM::constructFrom(exec, nfMesh, ofDivT);
        zero(nfDivT, 0.0);

        NeoN::TokenList scheme =
            NeoN::TokenList({std::string("bounded"), std::string("Gauss"), std::string("upwind")});
        fvcc::DivOperatorFactory<NeoN::scalar>::create(exec, nfMesh, scheme)
            ->div(nfDivT, nfPhi, nfT, dsl::Coeff(1.0));
        nfDivT.correctBoundaryConditions();

        REQUIRE_THAT(nfDivT, EqualsInternal(ofDivT, ApproxScalar(1e-11)));
    }

    auto ofU = randomVectorField(runTime, mesh, "U");
    auto nfU = NeoFOAM::constructFrom(exec, nfMesh, ofU);

    SECTION("GaussGreenDiv linearUpwind[vector] on " + execName)
    {
        Foam::IStringStream isLinUpwV("linearUpwind grad(U)");
        auto foamDivVec = Foam::fv::gaussConvectionScheme<Foam::vector>(mesh, ofPhi, isLinUpwV);
        Foam::volVectorField ofDivU("ofDivU", foamDivVec.fvcDiv(ofPhi, ofU));

        auto nfDivU = NeoFOAM::constructFrom(exec, nfMesh, ofDivU);
        zero(nfDivU, NeoN::Vec3(0.0, 0.0, 0.0));

        NeoN::TokenList scheme =
            NeoN::TokenList({std::string("linearUpwind"), std::string("grad(U)")});
        fvcc::GaussGreenDiv<NeoN::Vec3>(exec, nfMesh, scheme)
            .div(nfDivU, nfPhi, nfU, dsl::Coeff(1.0));
        nfDivU.correctBoundaryConditions();

        // The divergence values are O(1e6) (cells are ~2e-3 m, so 1/V is large), so a relative
        // tolerance per component is used rather than an absolute one.
        auto approxRel = [](NeoN::Vec3 nf, Foam::vector of)
        {
            return Catch::Approx(of[0]).epsilon(1e-6).margin(1.0) == nf[0]
                && Catch::Approx(of[1]).epsilon(1e-6).margin(1.0) == nf[1]
                && Catch::Approx(of[2]).epsilon(1e-6).margin(1.0) == nf[2];
        };
        REQUIRE_THAT(nfDivU, EqualsInternal(ofDivU, approxRel));
    }

    SECTION("GaussGreenDiv bounded[vector] on " + execName)
    {
        Foam::IStringStream isInnerVec("Gauss linearUpwind grad(U)");
        Foam::fv::boundedConvectionScheme<Foam::vector> foamBoundedDivVec(mesh, ofPhi, isInnerVec);
        Foam::volVectorField ofDivU("ofDivU", foamBoundedDivVec.fvcDiv(ofPhi, ofU));

        auto nfDivU = NeoFOAM::constructFrom(exec, nfMesh, ofDivU);
        zero(nfDivU, NeoN::Vec3(0.0, 0.0, 0.0));

        NeoN::TokenList scheme = NeoN::TokenList(
            {std::string("bounded"),
             std::string("Gauss"),
             std::string("linearUpwind"),
             std::string("grad(U)")}
        );
        fvcc::DivOperatorFactory<NeoN::Vec3>::create(exec, nfMesh, scheme)
            ->div(nfDivU, nfPhi, nfU, dsl::Coeff(1.0));
        nfDivU.correctBoundaryConditions();

        // Divergence values are O(1e6) here (see the linearUpwind[vector] section above), so a
        // relative tolerance per component is used rather than an absolute one.
        auto approxRelBounded = [](NeoN::Vec3 nf, Foam::vector of)
        {
            return Catch::Approx(of[0]).epsilon(1e-6).margin(1.0) == nf[0]
                && Catch::Approx(of[1]).epsilon(1e-6).margin(1.0) == nf[1]
                && Catch::Approx(of[2]).epsilon(1e-6).margin(1.0) == nf[2];
        };
        REQUIRE_THAT(nfDivU, EqualsInternal(ofDivU, approxRelBounded));
    }
}
