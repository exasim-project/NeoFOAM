// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoFOAM/NeoFOAM.hpp"

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "../test/common.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;
namespace dsl = NeoN::dsl;

#include "fvc.H"
#include "fvm.H"
#include "fvMatrices.H"

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

TEST_CASE("Poisson")
{
    Foam::Time& runTime = *timePtr;
    std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
    Foam::fvMesh& mesh = *meshPtr;

    auto ofU = nf::randomVectorField(runTime, mesh, "U");
    auto ofP = nf::randomScalarField(runTime, mesh, "p");
    auto ofPhi = nf::randDimField<Foam::surfaceScalarField>(mesh, {0, 3, -1, 0, 0}, "phi");
    auto ofGamma = nf::randDimField<Foam::surfaceScalarField>(mesh, {0, 2, -1, 0, 0}, "Gamma");

    SECTION("OpenFOAM")
    {
        SECTION("Poisson")
        {
            Foam::fvVectorMatrix ofUEqn(
                Foam::fvm::ddt(ofU) + Foam::fvm::div(ofPhi, ofU)
                - Foam::fvm::laplacian(ofGamma, ofU)
            );
            Foam::volScalarField rAU("rAU", 1.0 / ofUEqn.A());
            Foam::volVectorField HbyA("HbyA", rAU * ofUEqn.H());
            Foam::surfaceScalarField phiHbyA("phiHbyA", Foam::fvc::flux(HbyA));

            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::fvScalarMatrix pEqn(
                    Foam::fvm::laplacian(rAU, ofP) == Foam::fvc::div(phiHbyA)
                );

                pEqn.setReference(0, 0);
                return pEqn;
            };
        }
    }

    SECTION("NeoN")
    {
        auto [execName, exec] = GENERATE(allAvailableExecutor());
        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(schemesDict);
        auto& nfMesh = rt.mesh;
        auto& fieldCollection = fvcc::VectorCollection::instance(rt.db, "fieldCollection");
        auto& nfU = constructAndRegister(fieldCollection, rt, ofU);
        auto& nfPhi = nf::constructAndRegister(fieldCollection, rt, ofPhi);

        auto [nfP, nfGamma] = NeoFOAM::constFromMany(rt.exec, rt.nfMesh, ofP, ofGamma);

        SECTION(std::string("Poisson"))
        {
            nf::PDE<NeoN::Vec3> nfUEqn(
                dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfGamma, nfU),
                nfU,
                rt
            );
            nfUEqn.assemble();
            auto [crAU, hByA] = nf::computeRAUandHByA(nfUEqn);
            nf::constrainHbyA(nfU, nfP, hByA);
            const auto ddtScheme = nfUEqn.ddtScheme();
            nnfvcc::SurfaceField<NeoN::scalar> rAU = fvcc::SurfaceInterpolation<NeoN::scalar>(
                                                         rt.exec,
                                                         rt.nfMesh,
                                                         NeoN::TokenList({std::string("linear")})
            )
                                                         .interpolate(crAU);
            auto phiHbyA = nf::flux(hByA) + rAU * fvcc::ddtFluxCorr(nfU, nfPhi, rt.dt, ddtScheme);

            BENCHMARK(std::string(execName))
            {
                nf::PDE<NeoN::scalar> pEqn(
                    NeoN::dsl::imp::laplacian(rAU, nfP) - NeoN::dsl::exp::div(phiHbyA),
                    nfP,
                    rt
                );

                pEqn.setReference(0, 0);
                NeoN::fence(exec);
                return pEqn;
            };
        }
    }
}
