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

TEST_CASE("momentum")
{
    Foam::Time& runTime = *timePtr;
    std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
    Foam::fvMesh& mesh = *meshPtr;

    auto ofU = nf::randomVectorField(runTime, mesh, "U");
    auto ofP = nf::randomScalarField(runTime, mesh, "p");
    auto ofPhi = nf::randDimScalarField<Foam::surfaceScalarField>(mesh, {0, 3, -1, 0, 0}, "phi");
    auto ofGamma =
        nf::randDimScalarField<Foam::surfaceScalarField>(mesh, {0, 2, -1, 0, 0}, "Gamma");

    SECTION("OpenFOAM")
    {
        SECTION("without RHS")
        {
            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::fvVectorMatrix ofUEqn(
                    Foam::fvm::ddt(ofU) + Foam::fvm::div(ofPhi, ofU)
                    - Foam::fvm::laplacian(ofGamma, ofU)
                );
            };
        }
        SECTION("with RHS")
        {
            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::fvVectorMatrix ofUEqn(
                    Foam::fvm::ddt(ofU) + Foam::fvm::div(ofPhi, ofU)
                    - Foam::fvm::laplacian(ofGamma, ofU)
                );
                ofUEqn == Foam::fvc::grad(ofP);
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
        auto& nfU = constructFromVel(fieldCollection, rt, ofU);
        auto [nfP, nfPhi, nfGamma] =
            NeoFOAM::constFromMany(rt.exec, rt.nfMesh, ofP, ofPhi, ofGamma);

        SECTION(std::string("without RHS"))
        {
            BENCHMARK(std::string(execName))
            {
                nf::PDESolver<NeoN::Vec3> eqn(
                    dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU)
                        - dsl::imp::laplacian(nfGamma, nfU),
                    nfU,
                    rt
                );
                return eqn.assemble();
            };
        }

        SECTION(std::string("with RHS"))
        {
            BENCHMARK(std::string(execName))
            {
                nf::PDESolver<NeoN::Vec3> eqn(
                    dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU)
                        - dsl::imp::laplacian(nfGamma, nfU),
                    nfU,
                    rt
                );

                auto expr = dsl::Expression<NeoN::Vec3>(eqn.expression());
                auto ls = NeoN::la::LinearSystem<NeoN::Vec3, NeoN::localIdx>(eqn.linearSystem());
                expr.addOperator(-1.0 * dsl::exp::grad(nfP));
                eqn.assemble();
                return expr.assemble(rt.t, rt.dt, eqn.sparsityPattern(), ls);
            };
        }
    }
}
