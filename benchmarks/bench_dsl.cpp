// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "FoamAdapter/FoamAdapter.hpp"

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "common.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = FoamAdapter;
namespace dsl = NeoN::dsl;

#include "fvc.H"
#include "fvm.H"
#include "fvMatrices.H"

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

TEST_CASE("advection–diffusion-equation_scalar")
{
    Foam::Time& runTime = *timePtr;
    std::unique_ptr<Foam::fvMesh> meshPtr = FoamAdapter::createMesh(runTime);
    Foam::fvMesh& mesh = *meshPtr;

    auto ofT = randomScalarField(runTime, mesh, "T");
    Foam::surfaceScalarField ofPhi(
        Foam::IOobject("phi", "0", mesh, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
        mesh,
        Foam::dimensionedScalar("phi", Foam::dimensionSet(0, 3, -1, 0, 0), 0.0)
    );
    forAll(ofPhi, facei)
    {
        ofPhi[facei] = facei;
    }

    Foam::surfaceScalarField ofGamma(
        Foam::IOobject("Gamma", "0", mesh, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
        mesh,
        Foam::dimensionedScalar("phi", Foam::dimensionSet(0, 2, -1, 0, 0), 0.0)
    );

    SECTION("OpenFOAM")
    {
        SECTION("explicit-time-integration")
        {

            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::fvScalarMatrix advectDiffEqn(
                    Foam::fvm::ddt(ofT) + Foam::fvc::div(ofPhi, ofT)
                    - Foam::fvc::laplacian(ofGamma, ofT)
                );
                return advectDiffEqn;
            };
        }

        SECTION("implicit-time-integration")
        {
            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::fvScalarMatrix advectDiffEqn(
                    Foam::fvm::ddt(ofT) + Foam::fvm::div(ofPhi, ofT)
                    - Foam::fvm::laplacian(ofGamma, ofT)
                );
                return advectDiffEqn;
            };
        }
    }


    SECTION("NeoN")
    {
        auto [execName, exec] = GENERATE(allAvailableExecutor());

        auto rt = nf::createAdapterRunTime(runTime);

        auto& vectorCollection = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        fvcc::VolumeField<NeoN::scalar>& nfT =
            vectorCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::volScalarField> {
                    .exec = exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = ofT,
                    .name = "nfT"
                }
            );
        nfT.correctBoundaryConditions();

        auto& nfOldT = fvcc::oldTime(nfT);
        nfOldT.internalVector() = nfT.internalVector();

        auto nfPhi = nf::constructSurfaceField(rt.exec, rt.nfMesh, ofPhi);
        auto nfGamma = nf::constructSurfaceField(rt.exec, rt.nfMesh, ofGamma);

        SECTION(std::string("explicit-time-integration"))
        {
            rt.fvSchemesDict.insert(
                std::string("ddtSchemes"),
                NeoN::Dictionary(
                    {{std::string("type"), NeoN::TokenList({std::string("forwardEuler")})}}
                )
            );
            rt.fvSchemesDict.insert(
                std::string("divSchemes"),
                NeoN::Dictionary(
                    {{std::string("div(phi,nfT)"),
                      NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}}
                )
            );
            rt.fvSchemesDict.insert(
                std::string("laplacianSchemes"),
                NeoN::Dictionary(
                    {{std::string("laplacian(Gamma,nfT)"),
                      NeoN::TokenList(
                          {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                      )}}
                )
            );

            BENCHMARK(std::string(execName))
            {
                auto eqn = nf::PDESolver(
                    dsl::imp::ddt(nfT) + dsl::exp::div(nfPhi, nfT)
                        - dsl::exp::laplacian(nfGamma, nfT),
                    nfT,
                    rt
                );
                return eqn.assemble();
            };
        }

        SECTION(std::string("implicit-time-integration"))
        {
            rt.fvSchemesDict.insert(
                std::string("ddtSchemes"),
                NeoN::Dictionary(
                    {{std::string("type"), NeoN::TokenList({std::string("backwardEuler")})}}
                )
            );
            rt.fvSchemesDict.insert(
                std::string("divSchemes"),
                NeoN::Dictionary(
                    {{std::string("div(phi,nfT)"),
                      NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}}
                )
            );
            rt.fvSchemesDict.insert(
                std::string("laplacianSchemes"),
                NeoN::Dictionary(
                    {{std::string("laplacian(Gamma,nfT)"),
                      NeoN::TokenList(
                          {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                      )}}
                )
            );

            BENCHMARK(std::string(execName))
            {
                auto eqn = nf::PDESolver(
                    dsl::imp::ddt(nfT) + dsl::imp::div(nfPhi, nfT)
                        - dsl::imp::laplacian(nfGamma, nfT),
                    nfT,
                    rt
                );
                return eqn.assemble();
            };
        }
    }
}
