// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "../test/common.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;
namespace nf = NeoFOAM;

#include "fv.H"
#include "fvc.H"
#include "constrainHbyA.H"
#include "gaussGrad.H"
#include "gaussConvectionScheme.H"
#include "gaussLaplacianScheme.H"

TEST_CASE("pressureVelocityCoupling")
{
    Foam::Time& runTime = *timePtr;
    Foam::argList& args = *argsPtr;
    std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
    Foam::fvMesh& mesh = *meshPtr;

    auto ofU = nf::randomVectorField(runTime, mesh, "U");
    auto ofP = nf::randomScalarField(runTime, mesh, "p");
    auto ofPhi = nf::randDimScalarField<Foam::surfaceScalarField>(mesh, {0, 3, -1, 0, 0}, "phi");
    auto ofGamma =
        nf::randDimScalarField<Foam::surfaceScalarField>(mesh, {0, 2, -1, 0, 0}, "Gamma");

    SECTION("OpenFOAM")
    {
        Foam::fvVectorMatrix ofUEqn(
            Foam::fvm::ddt(ofU) + Foam::fvm::div(ofPhi, ofU) - Foam::fvm::laplacian(ofGamma, ofU)
        );

        SECTION("Compute rAU")
        {
            BENCHMARK("CPU") { return Foam::volScalarField("forAU", 1.0 / ofUEqn.A()); };
        }

        Foam::volScalarField forAU("forAU", 1.0 / ofUEqn.A());
        SECTION("Compute HbyA")
        {
            BENCHMARK("CPU") { return Foam::volVectorField("HbyA", forAU * ofUEqn.H()); };
        }

        Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());
        SECTION("constrainHbyA")
        {
            BENCHMARK("CPU")
            {
                Foam::constrainHbyA(forAU * ofUEqn.H(), ofU, ofP);
                return;
            };
        };
    }

    SECTION("NeoN")
    {
        auto [execName, exec] = GENERATE(allAvailableExecutor());
        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(schemesDict);
        auto& nfMesh = rt.mesh;
        auto& fieldCollection = fvcc::VectorCollection::instance(rt.db, "fieldCollection");
        auto& nfU = constructFromVel(fieldCollection, rt, ofU);
        auto [nfP, nfPhi, nfGamma] =
            NeoFOAM::constFromMany(rt.exec, rt.nfMesh, ofP, ofPhi, ofGamma);

        nf::PDESolver<NeoN::Vec3> nfUEqn(
            dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfGamma, nfU),
            nfU,
            rt
        );
        nfUEqn.assemble();

        SECTION("Compute rAU")
        {
            BENCHMARK(std::string(execName))
            {
                auto nfrAU = nf::computeRAU(nfUEqn);
                NeoN::fence(exec);
            };
        }

        auto nfrAU = nf::computeRAU(nfUEqn);
        SECTION("Compute HbyA")
        {
            BENCHMARK(std::string(execName))
            {
                nf::computeRAUandHByA(nfUEqn);
                NeoN::fence(exec);
                return;
            };
        }

        SECTION("constrainHbyA")
        {
            auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);
            BENCHMARK(std::string(execName))
            {
                nf::constrainHbyA(nfU, nfP, nfHbyA);

                NeoN::fence(exec);
                return;
            };
        }
    }
}
