// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "gaussConvectionScheme.H"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object


TEST_CASE("fvSolution")
{
    Foam::Time& runTime = *timePtr;
    Foam::argList& args = *argsPtr;

    std::string execName = "Serial";
    auto exec = NeoN::SerialExecutor {};

    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    NeoN::Dictionary fvSolutionDict = NeoFOAM::convert(mesh.solutionDict());
    NeoN::Dictionary& solverDict = fvSolutionDict.subDict("solvers");
    NeoN::Dictionary& solver1 = solverDict.subDict("T");

    SECTION("updateSolver")
    {
        SECTION("PCG")
        {
            solver1.insert("solver", std::string("PCG"));
            NeoFOAM::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Cg");
        }
        SECTION("PBiCG")
        {
            solver1.insert("solver", std::string("PBiCG"));
            NeoFOAM::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Bicg");
        }
        SECTION("PBiCGStab")
        {
            solver1.insert("solver", std::string("PBiCGStab"));
            NeoFOAM::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Bicgstab");
        }
        // GAMG has no dictionary-level mapping: Ginkgo's Multigrid needs mg_level /
        // coarse_solver entries this mapper cannot synthesise. It must be rejected here
        // rather than reaching Ginkgo, which fails with an opaque config error instead.
        SECTION("GAMG is rejected")
        {
            solver1.insert("solver", std::string("GAMG"));
            REQUIRE_THROWS_AS(NeoFOAM::updateSolver(solver1), std::runtime_error);
        }
    }

    SECTION("updatePreconditioner")
    {
        SECTION("diagonal")
        {
            solver1.insert("preconditioner", std::string("diagonal"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Jacobi");
            REQUIRE(preconditionerDict.get<int>("max_block_size") == 1);
        }
        SECTION("DIC")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ic");
        }
        SECTION("DILU")
        {
            solver1.insert("preconditioner", std::string("DILU"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ilu");
            REQUIRE(
                preconditionerDict.subDict("factorization").get<std::string>("type")
                == "factorization::ParIlu"
            );
        }
    }

    // The mapped dictionary carries a "reportName" label (Ginkgo
    // preconditioner+solver) that the per-solve residual report prints. These
    // run serially, so the serial (non-Schwarz) preconditioner names apply.
    SECTION("mapFvSolution reportName")
    {
        SECTION("DIC + PCG -> Ic+Cg")
        {
            solver1.insert("solver", std::string("PCG"));
            solver1.insert("preconditioner", std::string("DIC"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "Ic+Cg");
        }
        SECTION("DILU + PBiCGStab -> Ilu+Bicgstab")
        {
            solver1.insert("solver", std::string("PBiCGStab"));
            solver1.insert("preconditioner", std::string("DILU"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "Ilu+Bicgstab");
        }
        SECTION("diagonal + PCG -> Jacobi+Cg")
        {
            solver1.insert("solver", std::string("PCG"));
            solver1.insert("preconditioner", std::string("diagonal"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "Jacobi+Cg");
        }
        SECTION("configFile -> configFile")
        {
            solver1.insert("configFile", std::string("mySolver.json"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "configFile");
        }
    }
}
