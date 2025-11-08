// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 FoamAdapter authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <unordered_set>
#include <set>

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

    auto meshPtr = FoamAdapter::createMesh(exec, runTime);
    FoamAdapter::MeshAdapter& mesh = *meshPtr;

    NeoN::Dictionary fvSolutionDict = FoamAdapter::convert(mesh.solutionDict());
    NeoN::Dictionary& solverDict = fvSolutionDict.subDict("solvers");
    NeoN::Dictionary& solver1 = solverDict.subDict("solver1");

    SECTION("updateSolver")
    {
        SECTION("PCG")
        {
            solver1.insert("solver", std::string("PCG"));
            FoamAdapter::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Cg");
        }
        SECTION("PBiCG")
        {
            solver1.insert("solver", std::string("PBiCG"));
            FoamAdapter::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Bicg");
        }
        SECTION("PBiCGStab")
        {
            solver1.insert("solver", std::string("PBiCGStab"));
            FoamAdapter::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Bicgstab");
        }
    }

    SECTION("updatePreconditioner")
    {
        SECTION("DIC")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            FoamAdapter::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Jacobi");
            REQUIRE(preconditionerDict.get<int>("max_block_size") == 1);
        }
        SECTION("DILU")
        {
            solver1.insert("preconditioner", std::string("DILU"));
            FoamAdapter::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ilu");
            REQUIRE(preconditionerDict.get<bool>("reverse_apply") == false);
            REQUIRE(
                preconditionerDict.subDict("factorization").get<std::string>("type")
                == "factorization::ParIlu"
            );
        }
    }
}
