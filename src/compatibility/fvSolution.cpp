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

// int-tolerant readers: OpenFOAM commonly writes whole numbers as bare ints, which
// get<scalar>/get<label> would reject with bad_any_cast (no int coercion).
NeoN::scalar readScalarOr(const NeoN::Dictionary& d, const std::string& key, NeoN::scalar fallback)
{
    if (!d.contains(key)) return fallback;
    if (d.isType<int>(key)) return NeoN::scalar(d.get<int>(key));
    return d.get<NeoN::scalar>(key);
}

NeoN::label readLabelOr(const NeoN::Dictionary& d, const std::string& key, NeoN::label fallback)
{
    if (!d.contains(key)) return fallback;
    if (d.isType<int>(key)) return NeoN::label(d.get<int>(key));
    if (d.isType<NeoN::scalar>(key)) return NeoN::label(d.get<NeoN::scalar>(key));
    return d.get<NeoN::label>(key);
}

// Build the Ginkgo preconditioner that recovers OpenFOAM/SPUMA's twoStageGaussSeidel smoother: a
// forward Gauss-Seidel whose lower-triangular solve is approximated by `nInnerIter` damped
// (relaxation = omega) Jacobi iterations. Substituting a few Jacobi sweeps for the exact (and on a
// GPU inherently sequential) LowerTrs is the "two-stage" idea of Berger-Vergiat et al. 2021
// (DOI:10.48550/arXiv.2104.01196). Gauss-Seidel's `symmetric` defaults to false -> forward only.
NeoN::Dictionary
buildTwoStageGaussSeidel(NeoN::label nInnerIter, NeoN::scalar omega, bool distributed)
{
    // inner: nInnerIter damped-Jacobi sweeps approximating the triangular solve
    NeoN::Dictionary innerSolver(
        {{std::string("type"), std::string("solver::Ir")},
         {std::string("relaxation_factor"), omega},
         {std::string("solver"),
          NeoN::Dictionary(
              {{std::string("type"), std::string("preconditioner::Jacobi")},
               {std::string("max_block_size"), 1}}
          )},
         {std::string("criteria"),
          NeoN::Dictionary({{std::string("iteration"), static_cast<int>(nInnerIter)}})}}
    );

    NeoN::Dictionary gaussSeidel(
        {{std::string("type"), std::string("preconditioner::GaussSeidel")},
         {std::string("l_solver"), innerSolver}}
    );

    if (!distributed)
    {
        return gaussSeidel;
    }

    // Distributed: triangular solves cannot span ranks -> apply Gauss-Seidel per rank via additive
    // Schwarz (same pattern as the Ic/Ilu/Jacobi mappings in updatePreconditioner).
    return NeoN::Dictionary(
        {{std::string("type"), std::string("preconditioner::Schwarz")},
         {std::string("local_solver"), gaussSeidel}}
    );
}

// Map 'solver smoothSolver; smoother twoStageGaussSeidel;' to a Ginkgo IR(Richardson) driver over
// the two-stage Gauss-Seidel preconditioner. smoothSolver applies the smoother and checks the
// residual == preconditioned Richardson == solver::Ir (default relaxation 1.0) with the smoother as
// its inner solver. nInnerIter (default 1) and omega (default 0.9) mirror the OpenFOAM smoother
// controls; nSweeps is accepted but unused (the IR driver checks the residual every application).
NeoN::Dictionary mapTwoStageGaussSeidel(NeoN::Dictionary dict)
{
    NeoN::Logging::warn("Mapping smoothSolver/twoStageGaussSeidel to a Ginkgo IR(Gauss-Seidel) "
                        "two-stage smoother.\n");

    const NeoN::label nInnerIter = readLabelOr(dict, "nInnerIter", 1);
    const NeoN::scalar omega = readScalarOr(dict, "omega", 0.9);

    // Distributed detection mirrors updatePreconditioner (Schwarz wrap only for real MPI runs).
    NeoN::mpi::Environment mpiEnv;
    const bool distributed = mpiEnv.isInitialized() && mpiEnv.sizeRank() > 1;

    // Outer IR criteria from the OpenFOAM tolerance/relTol/maxIter controls.
    NeoFOAM::updateCriteria(dict);

    // Outer driver: solver::Ir (Richardson, default relaxation 1.0). Its inner "solver" is the
    // two-stage Gauss-Seidel preconditioner; a dictionary-valued "solver" also selects the Ginkgo
    // backend in SolverFactory::create.
    dict.insert("type", std::string("solver::Ir"));
    dict.insert("solver", buildTwoStageGaussSeidel(nInnerIter, omega, distributed));
    dict.insert("reportName", std::string("twoStageGaussSeidel"));

    // Keep only valid Ginkgo solver::Ir keys (plus the reportName meta key); drop every
    // OpenFOAM-only control (smoother, nSweeps, nInnerIter, omega, optimize, the original solver
    // string, ...) so Ginkgo's strict config check does not reject the configuration.
    const auto keep = [](const std::string& k)
    { return k == "type" || k == "solver" || k == "criteria" || k == "reportName"; };
    for (const auto& key : dict.keys())
    {
        if (!keep(key)) dict.remove(key);
    }
    return dict;
}

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

    // The twoStageGaussSeidel smoother is supported only with smoothSolver (-> IR/Richardson) or
    // GAMG (-> Multigrid). smoothSolver in turn supports only this smoother. Any other pairing is
    // rejected rather than silently degraded; use a configFile for arbitrary Ginkgo setups.
    const std::string solver = modSolverDict.isType<std::string>("solver")
                                   ? modSolverDict.get<std::string>("solver")
                                   : std::string();
    const std::string smoother = modSolverDict.isType<std::string>("smoother")
                                     ? modSolverDict.get<std::string>("smoother")
                                     : std::string();
    const bool twoStage = (smoother == "twoStageGaussSeidel");
    if (solver == "smoothSolver")
    {
        if (!twoStage)
        {
            throw std::runtime_error(
                "\nUnsupported smoothSolver smoother '" + smoother
                + "'.\nsmoothSolver is only supported with 'smoother twoStageGaussSeidel'.\n"
                  "Use a configFile for any other Ginkgo solver/smoother setup.\n"
            );
        }
        return mapTwoStageGaussSeidel(modSolverDict);
    }
    if (twoStage)
    {
        // GAMG + twoStageGaussSeidel (-> Ginkgo solver::Multigrid with Pgm coarsening and
        // IR(Gauss-Seidel) pre/post smoothers) is a planned follow-up; route there once
        // implemented. For now fail clearly rather than mis-mapping the GAMG controls.
        throw std::runtime_error(
            "\nThe twoStageGaussSeidel smoother is only supported with 'solver smoothSolver' or "
            "'solver GAMG'.\n"
            + (solver == "GAMG"
                   ? std::string("GAMG + twoStageGaussSeidel is not yet supported via a dictionary "
                                 "entry; use a configFile (solver::Multigrid) for now.\n")
                   : std::string("Got 'solver " + solver + "'.\n"))
        );
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
