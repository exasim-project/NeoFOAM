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
        // Reject before the rename below: solverName is a reference INTO the dictionary,
        // so assigning mapEntry->second.first to it would overwrite the OpenFOAM name
        // with "Ginkgo" and make this check unreachable for every solver.
        if (solverName == "GAMG")
        {
            throw std::runtime_error("\nGAMG Solver is not supported in NeoFOAM via dictionary "
                                     "entry, use configFile instead\n");
        }
        NeoN::Logging::warn("Replacing solver {} by {}", solverName, mapEntry->second.second);
        solverName = mapEntry->second.first;
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

    // A smoother (e.g. symGaussSeidel) with no explicit preconditioner maps the solver to BiCGStab
    // (see solverMap). BiCGStab is used for NON-symmetric systems such as the momentum matrix, so
    // the defaulted preconditioner must be valid for non-symmetric matrices. Incomplete-Cholesky
    // (DIC -> preconditioner::Ic) is symmetric-positive-definite only: on a non-SPD momentum block
    // its factorisation hits a negative pivot and SIGFPEs (sqrt of a negative diagonal). Default to
    // diagonal (block-Jacobi) instead, which is always defined and matches the intended behaviour.
    if (!solverDict.contains("preconditioner") && solverDict.contains("smoother"))
    {
        solverDict.insert("preconditioner", activeMap.at("diagonal"));
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

NeoN::Dictionary mapFvSolution(const NeoN::Dictionary& solverDict)
{
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

// Shared body for the relaxationFactors.<subDictName>.<field>[Final] lookup, used by both
// lookupEqnRelaxation ("equations") and lookupFieldRelaxation ("fields"). File-local /
// internal linkage only -> NOT declared in the header. Centralising the lookup here keeps the
// int-tolerant asScalar and isDict-guard fixes in ONE place, exercised by both
// the equations and the fields selectors.
//
// No relaxationFactors / <subDictName> sub-dict -> OpenFOAM "no entry, no relax" semantics:
// returns nullopt so the caller uses 1.0 (a no-op blend). The isDict guards keep this permissive:
// a malformed fvSolution where relaxationFactors (or the sub-dict) is present as a non-dictionary
// value degrades to nullopt instead of throwing bad_any_cast out of subDict.
//
// equations and fields are INDEPENDENT dicts (OpenFOAM honors both if both are set) -> no
// runtime guard couples them here; the no-double-relax convention is enforced by the caller
// + the field-relax test, not by code in this function.
static std::optional<NeoN::scalar> lookupRelaxation(
    const NeoN::Dictionary& fvSolution,
    const std::string& subDictName, // "equations" or "fields"
    const std::string& field,
    bool finalIter
)
{
    if (!fvSolution.contains("relaxationFactors") || !fvSolution.isDict("relaxationFactors"))
    {
        return std::nullopt;
    }
    const auto& rf = fvSolution.subDict("relaxationFactors");
    if (!rf.contains(subDictName) || !rf.isDict(subDictName))
    {
        return std::nullopt;
    }
    const auto& sub = rf.subDict(subDictName);

    // OpenFOAM fvSolution commonly writes whole-number relaxation factors as bare
    // integers (e.g. `UFinal 1;` / `pFinal 1;`), which `get<scalar>` would reject with
    // bad_any_cast (no int->scalar coercion). Mirror updateCriteria's extractScalar pattern:
    // coerce an int-typed entry to scalar, otherwise read it as scalar.
    auto asScalar = [](const NeoN::Dictionary& d, const std::string& k) -> NeoN::scalar
    { return d.isType<int>(k) ? NeoN::scalar(d.get<int>(k)) : d.get<NeoN::scalar>(k); };

    // Final-suffix selection: prefer <field>Final on the final iteration, then fall
    // back to the base <field> key (matches OpenFOAM's *Final relaxation convention).
    const std::string key = finalIter ? field + "Final" : field;
    if (sub.contains(key))
    {
        return asScalar(sub, key);
    }
    if (sub.contains(field))
    {
        return asScalar(sub, field);
    }
    return std::nullopt;
}

std::optional<NeoN::scalar>
lookupEqnRelaxation(const NeoN::Dictionary& fvSolution, const std::string& field, bool finalIter)
{
    return lookupRelaxation(fvSolution, "equations", field, finalIter);
}

std::optional<NeoN::scalar>
lookupFieldRelaxation(const NeoN::Dictionary& fvSolution, const std::string& field, bool finalIter)
{
    return lookupRelaxation(fvSolution, "fields", field, finalIter);
}

void createMappedFvSolutionDicts(RunTime& rt)
{
    auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    solverDict.subDict("p") = mapFvSolution(solverDict.subDict("p"));
    solverDict.subDict("U") = mapFvSolution(solverDict.subDict("U"));
    for (const std::string key : {"pFinal", "UFinal"})
    {
        if (solverDict.isDict(key))
        {
            solverDict.subDict(key) = mapFvSolution(solverDict.subDict(key));
        }
    }
}

} // namespace NeoFOAM
