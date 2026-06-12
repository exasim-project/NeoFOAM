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
            // Jacobi has no factorization, so the system must NOT be negated.
            REQUIRE_FALSE(solver1.contains("negateSystem"));
        }
        SECTION("DIC")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ic");
            REQUIRE(
                preconditionerDict.subDict("factorization").get<std::string>("type")
                == "factorization::ParIc"
            );
            // Incomplete Cholesky needs an SPD matrix -> request system negation.
            REQUIRE(solver1.get<bool>("negateSystem") == true);
            // No sweep count given -> Ginkgo's default (Auto) is used, no `iterations` key.
            REQUIRE_FALSE(preconditionerDict.subDict("factorization").contains("iterations"));
        }
        SECTION("DIC with nSweeps")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            solver1.insert("nSweeps", 3);
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ic");
            // nSweeps maps to the ParIc factorization's `iterations` and is consumed.
            REQUIRE(preconditionerDict.subDict("factorization").get<int>("iterations") == 3);
            REQUIRE(solver1.get<bool>("negateSystem") == true);
            REQUIRE_FALSE(solver1.contains("nSweeps"));
        }
        SECTION("DILU")
        {
            solver1.insert("preconditioner", std::string("DILU"));
            solver1.insert("nSweeps", 10);
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ilu");
            REQUIRE(
                preconditionerDict.subDict("factorization").get<std::string>("type")
                == "factorization::ParIlu"
            );
            REQUIRE(preconditionerDict.subDict("factorization").get<int>("iterations") == 10);
            REQUIRE(solver1.get<bool>("negateSystem") == true);
        }
        SECTION("DIC with lSolver isai")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            solver1.insert("lSolver", std::string("isai"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ic");
            // Ic applies the lower factor and its conj-transpose, so only an l_solver
            // override is injected, set to a lower-triangular ISAI approximate inverse.
            auto& lSolverDict = preconditionerDict.subDict("l_solver");
            REQUIRE(lSolverDict.get<std::string>("type") == "preconditioner::Isai");
            REQUIRE(lSolverDict.get<std::string>("isai_type") == "lower");
            // sparsity_power not requested -> Ginkgo default, no key injected.
            REQUIRE_FALSE(lSolverDict.contains("sparsity_power"));
            // The control key is consumed, not leaked into the Ginkgo config.
            REQUIRE_FALSE(solver1.contains("lSolver"));
        }
        SECTION("DIC with lSolver isai and sparsityPower")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            solver1.insert("lSolver", std::string("isai"));
            solver1.insert("sparsityPower", 2);
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            auto& lSolverDict = preconditionerDict.subDict("l_solver");
            REQUIRE(lSolverDict.get<std::string>("type") == "preconditioner::Isai");
            REQUIRE(lSolverDict.get<int>("sparsity_power") == 2);
            REQUIRE_FALSE(solver1.contains("sparsityPower"));
        }
        SECTION("DILU with lSolver isai injects both factors")
        {
            solver1.insert("preconditioner", std::string("DILU"));
            solver1.insert("lSolver", std::string("isai"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ilu");
            // Ilu is asymmetric: both lower and upper apply solvers are overridden.
            REQUIRE(preconditionerDict.subDict("l_solver").get<std::string>("isai_type") == "lower");
            REQUIRE(preconditionerDict.subDict("u_solver").get<std::string>("isai_type") == "upper");
        }
        SECTION("DIC with lSolver ir")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            solver1.insert("lSolver", std::string("ir"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ic");
            auto& lSolverDict = preconditionerDict.subDict("l_solver");
            REQUIRE(lSolverDict.get<std::string>("type") == "solver::Ir");
            // Fixed-iteration apply (never residual-stopped): default 1 sweep == aDIC.
            // Ir needs Ginkgo's explicit criterion form {type: Iteration, max_iters: N}.
            REQUIRE(lSolverDict.subDict("criteria").get<std::string>("type") == "Iteration");
            REQUIRE(lSolverDict.subDict("criteria").get<int>("max_iters") == 1);
            // Inner relaxation is a point-Jacobi sweep.
            REQUIRE(lSolverDict.subDict("solver").get<std::string>("type") == "preconditioner::Jacobi");
            // No relaxation factor requested -> Ginkgo default, no key injected.
            REQUIRE_FALSE(lSolverDict.contains("relaxation_factor"));
        }
        SECTION("DIC with lSolver ir, sweeps and relaxationFactor")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            solver1.insert("lSolver", std::string("ir"));
            solver1.insert("lSolverSweeps", 3);
            solver1.insert("relaxationFactor", NeoN::scalar(0.8));
            NeoFOAM::updatePreconditioner(solver1);
            auto& lSolverDict = solver1.subDict("preconditioner").subDict("l_solver");
            REQUIRE(lSolverDict.subDict("criteria").get<int>("max_iters") == 3);
            REQUIRE(lSolverDict.get<NeoN::scalar>("relaxation_factor") == 0.8);
            REQUIRE_FALSE(solver1.contains("lSolverSweeps"));
            REQUIRE_FALSE(solver1.contains("relaxationFactor"));
        }
        SECTION("DILU with lSolver ir injects both factors")
        {
            solver1.insert("preconditioner", std::string("DILU"));
            solver1.insert("lSolver", std::string("ir"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ilu");
            REQUIRE(preconditionerDict.subDict("l_solver").get<std::string>("type") == "solver::Ir");
            REQUIRE(preconditionerDict.subDict("u_solver").get<std::string>("type") == "solver::Ir");
        }
        SECTION("lSolver isai on a non-factorization preconditioner throws")
        {
            solver1.insert("preconditioner", std::string("diagonal"));
            solver1.insert("lSolver", std::string("isai"));
            REQUIRE_THROWS(NeoFOAM::updatePreconditioner(solver1));
        }
        SECTION("lSolver ir on a non-factorization preconditioner throws")
        {
            solver1.insert("preconditioner", std::string("diagonal"));
            solver1.insert("lSolver", std::string("ir"));
            REQUIRE_THROWS(NeoFOAM::updatePreconditioner(solver1));
        }
        SECTION("unknown lSolver value throws")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            solver1.insert("lSolver", std::string("bogus"));
            REQUIRE_THROWS(NeoFOAM::updatePreconditioner(solver1));
        }
        SECTION("aDIC maps to a native marker + negateSystem")
        {
            solver1.insert("preconditioner", std::string("aDIC"));
            NeoFOAM::updatePreconditioner(solver1);
            // aDIC becomes a {type: aDIC} marker dict that GinkgoSolver injects as a generated
            // preconditioner; it needs an SPD matrix so the system is negated.
            REQUIRE(solver1.isDict("preconditioner"));
            REQUIRE(solver1.subDict("preconditioner").get<std::string>("type") == "aDIC");
            REQUIRE(solver1.get<bool>("negateSystem") == true);
        }
        SECTION("preconReuse is normalized to int and kept for GinkgoSolver")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            solver1.insert("preconReuse", 5);
            NeoFOAM::updatePreconditioner(solver1);
            // preconReuse is not consumed here (GinkgoSolver reads it), only normalized.
            REQUIRE(solver1.contains("preconReuse"));
            REQUIRE(solver1.get<int>("preconReuse") == 5);
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
        SECTION("DIC + lSolver isai + PCG -> Ic(Isai)+Cg")
        {
            solver1.insert("solver", std::string("PCG"));
            solver1.insert("preconditioner", std::string("DIC"));
            solver1.insert("lSolver", std::string("isai"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "Ic(Isai)+Cg");
        }
        SECTION("DIC + lSolver ir + PCG -> Ic(Ir)+Cg")
        {
            solver1.insert("solver", std::string("PCG"));
            solver1.insert("preconditioner", std::string("DIC"));
            solver1.insert("lSolver", std::string("ir"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "Ic(Ir)+Cg");
        }
        SECTION("aDIC + PCG -> aDIC+Cg")
        {
            solver1.insert("solver", std::string("PCG"));
            solver1.insert("preconditioner", std::string("aDIC"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "aDIC+Cg");
        }
        SECTION("configFile -> configFile")
        {
            solver1.insert("configFile", std::string("mySolver.json"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "configFile");
        }
    }
}
