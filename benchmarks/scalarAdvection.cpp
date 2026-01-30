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

TEST_CASE("scalarAdvection")
{
    Foam::Time& runTime = *timePtr;
    std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
    Foam::fvMesh& mesh = *meshPtr;

    auto ofT = nf::randomScalarField(mesh, "T");
    auto ofPhi = nf::randDimScalarField<Foam::surfaceScalarField>(mesh, {0, 3, -1, 0, 0}, "phi");
    auto ofGamma =
        nf::randDimScalarField<Foam::surfaceScalarField>(mesh, {0, 2, -1, 0, 0}, "Gamma");

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
        auto rt = nf::createAdapterRunTime(runTime, exec);

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
        fvcc::rotateOldTimes(nfT);

        auto [nfPhi, nfGamma] = NeoFOAM::constFromMany(rt.exec, rt.nfMesh, ofPhi, ofGamma);

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

        SECTION(std::string("explicit-time-integration"))
        {
            rt.fvSchemesDict.insert(
                std::string("ddtSchemes"),
                NeoN::Dictionary({{std::string("ddt(nfT)"), {std::string("BDF1")}}})
            );

            BENCHMARK(std::string(execName))
            {
                auto eqn = nf::PDESolver(
                    dsl::imp::ddt(nfT) + dsl::exp::div(nfPhi, nfT)
                        - dsl::exp::laplacian(nfGamma, nfT),
                    nfT,
                    rt
                );
                eqn.assemble();
                NeoN::fence(exec);
            };
        }

        SECTION(std::string("implicit-time-integration"))
        {
            rt.fvSchemesDict.insert(
                std::string("ddtSchemes"),
                NeoN::Dictionary({{std::string("ddt(nfT)"), {std::string("BDF2")}}})
            );

            BENCHMARK(std::string(execName))
            {
                auto eqn = nf::PDESolver(
                    dsl::imp::ddt(nfT) + dsl::imp::div(nfPhi, nfT)
                        - dsl::imp::laplacian(nfGamma, nfT),
                    nfT,
                    rt
                );
                eqn.assemble();
                NeoN::fence(exec);
            };
        }
    }
}
