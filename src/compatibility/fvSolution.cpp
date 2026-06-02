// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
/* This file implements comparison operator to compare OpenFOAM and corresponding NeoFOAM fields
 * TODO the comparison operator only make sense for testing purposes
 * so this should be part of the tests
 */


#include "NeoFOAM/compatibility/fvSolution.hpp"

#include <map>

#include <NeoN/core/logging.hpp>
#include <NeoN/core/mpi/environment.hpp>
#include <NeoN/core/primitives/scalar.hpp>
#include <NeoN/core/primitives/label.hpp>


namespace NeoFOAM
{

void updateSolver(NeoN::Dictionary& solverDict)
{
    // Map OpenFOAM solver names to NeoN/Ginkgo solver names and types
    static const std::map<std::string, std::pair<std::string, std::string>> solverMap = {
        {"PCG", {"Ginkgo", "solver::Cg"}},
        {"PBiCG", {"Ginkgo", "solver::Bicg"}},
        {"PBiCGStab", {"Ginkgo", "solver::Bicgstab"}},
        {"smoothSolver", {"Ginkgo", "solver::Bicgstab"}},
        {"GAMG", {"Ginkgo", "solver::Multigrid"}},
    };

    std::string& solverName = solverDict.get<std::string>("solver");
    auto mapEntry = solverMap.find(solverName);
    if (mapEntry != solverMap.end())
    {
        NeoN::Logging::warn("Replacing solver {} by {}", solverName, mapEntry->second.second);
        solverName = mapEntry->second.first;
        if (solverName == "GAMG")
        {
            throw std::runtime_error("\nGAMG Solver is not supported in NeoFOAM via dictionary "
                                     "entry, use configFile instead\n");
        }
        solverDict.insert("type", mapEntry->second.second);
    }
}

void updatePreconditioner(NeoN::Dictionary& solverDict)
{
    // Map OpenFOAM preconditioner types to NeoN/Ginkgo preconditioner types (single rank).
    static const std::map<std::string, NeoN::Dictionary> preconditionerMap = {
        {"DIC",
         NeoN::Dictionary(
             {{std::string("type"), std::string("preconditioner::Ic")},
              {std::string("factorization"),
               NeoN::Dictionary({{std::string("type"), std::string("factorization::ParIc")}})}}
         )},
        {"diagonal",
         NeoN::Dictionary(
             {{std::string("type"), std::string("preconditioner::Jacobi")},
              {std::string("max_block_size"), 1}}
         )},
        {"DILU",
         NeoN::Dictionary(
             {{std::string("type"), std::string("preconditioner::Ilu")},
              {std::string("factorization"),
               NeoN::Dictionary({{std::string("type"), std::string("factorization::ParIlu")}})}}
         )}
    };

    // Map OpenFOAM preconditioner types to a Ginkgo additive Schwarz wrapper whose
    // local_solver is the per-rank preconditioner. Required in distributed runs because
    // factorisation-based preconditioners (Ic/Ilu) and Jacobi cannot span ranks; Schwarz
    // applies the local_solver to each rank's local block.
    static const std::map<std::string, NeoN::Dictionary> distributedPreconditionerMap = {
        {"DIC",
         NeoN::Dictionary(
             {{std::string("type"), std::string("preconditioner::Schwarz")},
              {std::string("local_solver"),
               NeoN::Dictionary(
                   {{std::string("type"), std::string("preconditioner::Ic")},
                    {std::string("factorization"),
                     NeoN::Dictionary({{std::string("type"), std::string("factorization::ParIc")}})}
                   }
               )}}
         )},
        {"diagonal",
         NeoN::Dictionary(
             {{std::string("type"), std::string("preconditioner::Schwarz")},
              {std::string("local_solver"),
               NeoN::Dictionary(
                   {{std::string("type"), std::string("preconditioner::Jacobi")},
                    {std::string("max_block_size"), 1}}
               )}}
         )},
        {"DILU",
         NeoN::Dictionary(
             {{std::string("type"), std::string("preconditioner::Schwarz")},
              {std::string("local_solver"),
               NeoN::Dictionary(
                   {{std::string("type"), std::string("preconditioner::Ilu")},
                    {std::string("factorization"),
                     NeoN::Dictionary({{std::string("type"), std::string("factorization::ParIlu")}})
                    }}
               )}}
         )}
    };

    // Distributed-mode detection: `Environment::sizeRank()` returns
    // static_cast<size_t>(-1) when MPI hasn't been initialised (a serial run, or
    // a parallel binary that hasn't called MPI_Init yet). Comparing > 1 alone
    // would be true in those cases too, so we'd wrap the preconditioner in
    // Schwarz for a serial solve and SIGILL inside Ginkgo. Gate the wrap on
    // MPI being initialised AND sizeRank > 1.
    NeoN::mpi::Environment mpiEnv;
    const bool distributed = mpiEnv.isInitialized() && mpiEnv.sizeRank() > 1;
    const auto& activeMap = distributed ? distributedPreconditionerMap : preconditionerMap;

    // if no preconditioner is set but smoother switch to BiCGStab with BJ
    if (!solverDict.contains("preconditioner") && solverDict.contains("smoother"))
    {
        solverDict.insert("preconditioner", activeMap.at("DIC"));
    }

    if (solverDict.contains("smoother"))
    {
        solverDict.remove("smoother");
    }

    if (!solverDict.isDict("preconditioner"))
    {
        std::string& preconditionerName = solverDict.get<std::string>("preconditioner");

        // pop preconditioner if none and early return
        if (preconditionerName == "none")
        {
            solverDict.remove("preconditioner");
            return;
        }

        if (preconditionerName == "GAMG")
        {
            throw std::runtime_error("\nGAMG Preconditioner is not supported in NeoFOAM via "
                                     "dictionary entry, use a configFile instead\n");
        }

        auto mapEntry = activeMap.find(preconditionerName);
        if (mapEntry != activeMap.end())
        {
            NeoN::Logging::warn(
                "Replacing preconditioner {} by {}{}",
                preconditionerName,
                mapEntry->second.get<std::string>("type"),
                distributed ? " (Schwarz-wrapped for distributed run)" : ""
            );
            solverDict.insert("preconditioner", mapEntry->second);
        }
    }
}

void updateCriteria(NeoN::Dictionary& solverDict)
{
    // parse given dictionary, get numeric value of key in a safe way
    auto extractScalar = [](NeoN::Dictionary& d, std::string key)
    {
        NeoN::scalar ret =
            (d.isType<int>(key)) ? NeoN::scalar(d.get<int>(key)) : d.get<NeoN::scalar>(key);
        d.remove(key);
        return ret;
    };

    // Ensure the criteria dictionary exists
    if (!solverDict.contains("criteria"))
    {
        solverDict.insert("criteria", NeoN::Dictionary());
    }

    // set default max iteration count
    {
        NeoN::Dictionary& criteriaDict = solverDict.subDict("criteria");
        criteriaDict.insert("iteration", 1000);
    }

    // Set default values for relative residual norm and iteration count
    if (solverDict.contains("relTol"))
    {
        NeoN::Dictionary& criteriaDict = solverDict.subDict("criteria");
        criteriaDict.insert("initial_residual_norm", extractScalar(solverDict, "relTol"));
    }
    if (solverDict.contains("maxIter"))
    {
        NeoN::Dictionary& criteriaDict = solverDict.subDict("criteria");
        criteriaDict.insert("iteration", solverDict.get<NeoN::label>("maxIter"));
        solverDict.remove("maxIter");
    }
    if (solverDict.contains("tolerance"))
    {
        NeoN::Dictionary& criteriaDict = solverDict.subDict("criteria");
        criteriaDict.insert("absolute_residual_norm", extractScalar(solverDict, "tolerance"));
    }

    NeoN::Dictionary& criteriaDict = solverDict.subDict("criteria");
}


namespace
{

// Strip a Ginkgo "namespace::Name" identifier down to "Name" (e.g.
// "solver::Cg" -> "Cg", "preconditioner::Ic" -> "Ic").
std::string stripNamespace(const std::string& s)
{
    const auto pos = s.rfind("::");
    return pos == std::string::npos ? s : s.substr(pos + 2);
}

// Build a label from the Ginkgo solver/preconditioner ACTUALLY selected after
// mapping (e.g. "Ic+Cg", or "Schwarz(Ic)+Cg" for a distributed run). Reads the
// post-mapping dictionary, so it reflects exactly what NeoN/Ginkgo will run.
std::string ginkgoSolverLabel(const NeoN::Dictionary& mapped)
{
    if (mapped.contains("configFile")) return "configFile";

    std::string solver =
        mapped.contains("type") ? stripNamespace(mapped.get<std::string>("type")) : "Ginkgo";

    std::string precond;
    if (mapped.contains("preconditioner"))
    {
        if (mapped.isDict("preconditioner"))
        {
            const NeoN::Dictionary& pd = mapped.subDict("preconditioner");
            if (pd.contains("type"))
            {
                const std::string ptype = pd.get<std::string>("type");
                precond = stripNamespace(ptype);
                // Unwrap the additive-Schwarz local solver so the meaningful
                // per-rank preconditioner is visible in distributed runs.
                if (ptype == "preconditioner::Schwarz" && pd.contains("local_solver")
                    && pd.isDict("local_solver"))
                {
                    const NeoN::Dictionary& ld = pd.subDict("local_solver");
                    if (ld.contains("type"))
                        precond += "(" + stripNamespace(ld.get<std::string>("type")) + ")";
                }
            }
        }
        else
        {
            precond = stripNamespace(mapped.get<std::string>("preconditioner"));
        }
    }
    return precond.empty() ? solver : precond + "+" + solver;
}

} // namespace

NeoN::Dictionary mapFvSolution(const NeoN::Dictionary& solverDict, const std::string& fieldName)
{
    (void)fieldName; // residual reporting uses the field name carried by the solver
    NeoN::Dictionary modSolverDict = solverDict;

    if (solverDict.contains("configFile"))
    {
        modSolverDict.insert("reportName", std::string("configFile"));
        return modSolverDict;
    }

    NeoN::Logging::warn("Mapping OpenFOAM solver settings to NeoN settings.\n"
                        "Currently, it is advisable to specify a configFile\n"
                        "for fine grained Ginkgo solver control\n");
    updateSolver(modSolverDict);
    updatePreconditioner(modSolverDict);
    updateCriteria(modSolverDict);

    // Stash the Ginkgo solver/preconditioner label (e.g. "Ic+Cg") that fvSolution
    // mapped to, for the per-solve residual report. Read from the post-mapping
    // dict so it reflects what Ginkgo runs; the Ginkgo backend's parse() ignores
    // this meta key. Computed once here -> no per-solve / hot-path cost.
    modSolverDict.insert("reportName", ginkgoSolverLabel(modSolverDict));

    return modSolverDict;
}

} // namespace NeoFOAM
