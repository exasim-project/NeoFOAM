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
    NeoN::UmpireMempoolHandler::setupUmpirePool(NeoN::MemorySpace::GPU, 1400 * mesh.nCells());

    auto ofU = nf::randomVectorField(runTime, mesh, "U");
    auto ofP = nf::randomScalarField(runTime, mesh, "p");
    auto ofPhi = nf::randDimField<Foam::surfaceScalarField>(mesh, {0, 3, -1, 0, 0}, "phi");
    auto ofGamma = nf::randDimField<Foam::surfaceScalarField>(mesh, {0, 2, -1, 0, 0}, "Gamma");

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
        auto& nfMesh = rt.nfMesh;
        auto& fieldCollection = fvcc::VectorCollection::instance(rt.db, "fieldCollection");
        auto& nfU = constructAndRegister(fieldCollection, rt, ofU);
        auto [nfP, nfPhi, nfGamma] =
            NeoFOAM::constFromMany(rt.exec, rt.nfMesh, ofP, ofPhi, ofGamma);

        SECTION(std::string("without RHS"))
        {
                nf::PDESolver<NeoN::Vec3> eqn(
                    dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU)
                        - dsl::imp::laplacian(nfGamma, nfU),
                    nfU,
                    rt
                );
            BENCHMARK(std::string(execName))
            {
                eqn.assemble();
                NeoN::fence(exec);
                return;
            };
        }


        SECTION(std::string("without RHS optimized cellbased"))
        {

        nf::PDESolver<NeoN::Vec3> eqn(
            dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU)
                - dsl::imp::laplacian(nfGamma, nfU),
            nfU,
            rt
        );

        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        auto lsOpt = NeoN::la::createEmptyLinearSystem<NeoN::Vec3>(nfMesh, cellIterator);
        auto [sp, mi] = NeoN::la::createSparsityPatternFaceToMatrixAddress<NeoN::la::CsrSparsityPattern<NeoN::localIdx>>(nfMesh);
        cellIterator->setComputeCellBasedData(nfMesh, sp, mi);

	    auto opts = std::vector<std::shared_ptr<NeoN::dsl::Optimizer<NeoN::dsl::Expression<Vec3>>>> {
		std::make_shared<NeoN::dsl::DivLapOptimizer<NeoN::dsl::Expression<Vec3>>>()
	    };

            BENCHMARK(std::string(execName))
            {
                eqn.assemble2(lsOpt, opts );
                NeoN::fence(exec);
                return;
            };
        }

        SECTION(std::string("without RHS optimized"))
        {
                nf::PDESolver<NeoN::Vec3> eqn(
                    dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU)
                        - dsl::imp::laplacian(nfGamma, nfU),
                    nfU,
                    rt
                );

        auto cellIterator = std::make_shared<NeoN::la::FaceBasedIterator>();
        auto lsOpt = NeoN::la::createEmptyLinearSystem<NeoN::Vec3>(nfMesh, cellIterator);

            BENCHMARK(std::string(execName))
            {
                eqn.assemble2(lsOpt,true);
                NeoN::fence(exec);
                return;
            };
        }


//       SECTION(std::string("with RHS"))
//       {
//           BENCHMARK(std::string(execName))
//           {
//               nf::PDESolver<NeoN::Vec3> eqn(
//                   dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU)
//                       - dsl::imp::laplacian(nfGamma, nfU),
//                   nfU,
//                   rt
//               );
//
//               auto expr = dsl::Expression<NeoN::Vec3>(eqn.expression());
//               auto ls = NeoN::la::LinearSystem<
//                   NeoN::Vec3,
//                   NeoN::la::CSRMatrix<NeoN::Vec3, NeoN::localIdx>>(eqn.linearSystem());
//               expr.addOperator(-1.0 * dsl::exp::grad(nfP));
//               eqn.assemble();
//               expr.assemble(rt.t, rt.dt, ls);
//               NeoN::fence(exec);
//               return;
//           };
//       }
    }
}
