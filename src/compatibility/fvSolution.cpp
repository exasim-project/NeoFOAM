// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
/* This file implements comparison operator to compare OpenFOAM and corresponding NeoFOAM fields
 * TODO the comparison operator only make sense for testing purposes
 * so this should be part of the tests
 */


#include "NeoFOAM/compatibility/fvSolution.hpp"

#include <algorithm>
#include <map>
#include <set>

#include <NeoN/core/logging.hpp>
#include <NeoN/core/mpi/environment.hpp>
#include <NeoN/core/primitives/scalar.hpp>
#include <NeoN/core/primitives/label.hpp>

#include "regExp.H"


namespace NeoFOAM
{

namespace
{

// The dictionary keys, sorted — the NeoN dictionary is a hash map, so the raw order
// would make the error messages differ from run to run.
std::vector<std::string> sortedKeys(const NeoN::Dictionary& dict)
{
    std::vector<std::string> keys = dict.keys();
    std::sort(keys.begin(), keys.end());
    return keys;
}

std::string join(const std::vector<std::string>& items)
{
    std::string joined;
    for (const auto& item : items)
    {
        if (!joined.empty()) joined += ", ";
        joined += item;
    }
    return joined;
}

} // namespace

FvSolutionKeyNotFound::FvSolutionKeyNotFound(
    const std::string& field,
    const std::string& dictName,
    const std::vector<std::string>& availableKeys
)
    : std::runtime_error(
        "No entry for field '" + field + "' in " + dictName
        + "; available keys: " + join(availableKeys)
    )
    , field_(field)
    , dictName_(dictName)
    , availableKeys_(availableKeys)
{}

bool keyMatches(const std::string& key, const std::string& name)
{
    if (key == name) return true;

    // Only a quoted keyword is a regular expression (see NeoFOAM::dictKey); the pattern
    // must match the whole name, as Foam::dictionary's pattern lookup does.
    if (key.size() < 2 || key.front() != '"' || key.back() != '"') return false;
    return Foam::regExp(key.substr(1, key.size() - 2)).match(name);
}

std::optional<std::string> matchKey(const NeoN::Dictionary& dict, const std::string& name)
{
    if (dict.contains(name)) return name;

    std::vector<std::string> matches;
    for (const auto& key : sortedKeys(dict))
    {
        if (keyMatches(key, name)) matches.push_back(key);
    }

    if (matches.empty()) return std::nullopt;
    if (matches.size() > 1)
    {
        throw std::runtime_error(
            "Several fvSolution keys match the field '" + name + "': " + join(matches)
            + "\nOpenFOAM resolves this by file order, which the NeoN dictionary does not "
              "keep - spell out the entry for '"
            + name + "' instead."
        );
    }
    return matches.front();
}

bool hasSolverSettings(const NeoN::Dictionary& solvers, const std::string& field)
{
    const auto key = matchKey(solvers, field);
    return key && solvers.isDict(*key);
}

NeoN::Dictionary& solverSettings(NeoN::Dictionary& solvers, const std::string& field)
{
    const auto key = matchKey(solvers, field);
    if (!key)
    {
        throw FvSolutionKeyNotFound(field, "system/fvSolution/solvers", sortedKeys(solvers));
    }
    return solvers.subDict(*key);
}

const NeoN::Dictionary& solverSettings(const NeoN::Dictionary& solvers, const std::string& field)
{
    const auto key = matchKey(solvers, field);
    if (!key)
    {
        throw FvSolutionKeyNotFound(field, "system/fvSolution/solvers", sortedKeys(solvers));
    }
    return solvers.subDict(*key);
}

void mapSolverSettings(NeoN::Dictionary& solvers, const std::string& field)
{
    if (!hasSolverSettings(solvers, field)) return;
    NeoN::Dictionary& settings = solverSettings(solvers, field);
    settings = mapFvSolution(settings);
}

namespace
{

// Distributed-mode detection: `Environment::sizeRank()` returns
// static_cast<size_t>(-1) when MPI hasn't been initialised (a serial run, or
// a parallel binary that hasn't called MPI_Init yet). Comparing > 1 alone
// would be true in those cases too, so we'd wrap the preconditioner in
// Schwarz for a serial solve and SIGILL inside Ginkgo. Gate the wrap on
// MPI being initialised AND sizeRank > 1.
bool isDistributedRun()
{
    NeoN::mpi::Environment mpiEnv;
    return mpiEnv.isInitialized() && mpiEnv.sizeRank() > 1;
}

// Build an explicit Ginkgo Iteration stopping criterion: {type Iteration; max_iters N}.
// The multigrid solver parses its `criteria` via parse_or_get_factory_vector, which
// requires the explicit factory form (a `type` key) - the minimal {iteration N} form
// accepted by Cg/Ir's parse_or_get_criteria is NOT understood there and aborts with
// "Contains empty, but try to get string". Use the explicit form everywhere inside the
// multigrid config so it is valid for every consumer.
NeoN::Dictionary makeIterationCriterion(int maxIters)
{
    NeoN::Dictionary crit;
    crit.insert("type", std::string("Iteration"));
    crit.insert("max_iters", maxIters);
    return crit;
}

// Build a Jacobi-preconditioned smoother applying @p nSweeps sweeps per invocation
// (per multigrid level).
//
// The textbook choice would be Ginkgo's solver::Ir (Richardson) around a point-Jacobi,
// but Ir names its inner operator with the key `solver`, and NeoN's dictionary->pnode
// conversion any_casts every `solver` entry to std::string to strip its own
// `solver Ginkgo;` marker - a sub-dictionary there aborts the solve with bad_any_cast.
// Ginkgo's Chebyshev needs its `foci` as a two-element array, which the NeoN dictionary
// cannot express either. solver::Cg takes its inner operator under `preconditioner`,
// runs exactly the requested number of sweeps via the Iteration criterion, and its
// single-sweep form is Jacobi with the optimal (Rayleigh-quotient) step length - the
// scaled correction that makes a Jacobi smoother competitive with OpenFOAM's GAMG.
//
// In a distributed run the point-Jacobi is wrapped in an additive Schwarz whose
// local_solver it is - the same reason plain Jacobi/Ic/Ilu are Schwarz-wrapped in
// distributedPreconditionerMap: a point preconditioner cannot span MPI ranks.
NeoN::Dictionary makeJacobiSmoother(int nSweeps, bool distributed)
{
    NeoN::Dictionary jacobi;
    jacobi.insert("type", std::string("preconditioner::Jacobi"));
    jacobi.insert("max_block_size", 1);

    NeoN::Dictionary preconditioner = jacobi;
    if (distributed)
    {
        preconditioner = NeoN::Dictionary();
        preconditioner.insert("type", std::string("preconditioner::Schwarz"));
        preconditioner.insert("local_solver", jacobi);
    }

    NeoN::Dictionary smoother;
    smoother.insert("type", std::string("solver::Cg"));
    smoother.insert("preconditioner", preconditioner);
    smoother.insert("criteria", makeIterationCriterion(nSweeps));
    return smoother;
}

// Build a Ginkgo algebraic-multigrid factory dictionary - the NeoN analogue of OpenFOAM's
// GAMG. Coarsening is Pgm (parallel graph match aggregation, Ginkgo's built-in AMG
// coarsening); smoother and coarsest-level solver are Jacobi-preconditioned (see
// makeJacobiSmoother). Fixed defaults follow Ginkgo's shipped pgm-multigrid-cg.json:
//   - V-cycle, up to 10 levels,
//   - 8 smoother sweeps on the coarsest level for a reasonably converged coarse solve,
//   - a single V-cycle per application, which is how the multigrid is used both as the
//     GAMG solver (CG-accelerated) and as a GAMG preconditioner.
// Ginkgo's post smoother is the pre smoother (post_uses_pre defaults to true): turning
// that off takes a boolean, and NeoN's dictionary->pnode conversion has no boolean type
// (it aborts with "unsupported type"), so the sweep count is shared - see
// makeGamgReplacement's handling of nPostSweeps.
NeoN::Dictionary makeMultigridDict(int minCoarseRows, int nSweeps, bool distributed)
{
    NeoN::Dictionary mgLevel;
    mgLevel.insert("type", std::string("multigrid::Pgm"));

    NeoN::Dictionary mg;
    mg.insert("type", std::string("solver::Multigrid"));
    mg.insert("max_levels", 10);
    mg.insert("min_coarse_rows", minCoarseRows);
    mg.insert("cycle", std::string("v"));
    mg.insert("mg_level", mgLevel);
    mg.insert("pre_smoother", makeJacobiSmoother(nSweeps, distributed));
    mg.insert("coarsest_solver", makeJacobiSmoother(8, distributed));
    mg.insert("default_initial_guess", std::string("zero"));
    mg.insert("criteria", makeIterationCriterion(1));
    return mg;
}

// Read and remove an OpenFOAM count entry. OpenFOAM writes these as bare integers, but
// a dictionary that went through an expansion can carry them as scalars - mirror
// updateCriteria's extractScalar and accept both rather than throwing bad_any_cast.
int consumeCount(NeoN::Dictionary& dict, const std::string& key)
{
    const int value =
        dict.isType<int>(key) ? dict.get<int>(key) : static_cast<int>(dict.get<NeoN::scalar>(key));
    dict.remove(key);
    return value;
}

// Build the Ginkgo multigrid that replaces OpenFOAM's GAMG, consuming the GAMG tuning
// keys of @p solverDict as it goes.
//
// Everything the mapping does not understand has to be removed: NeoN's parse() only
// strips solver/coupled/reportName/l1ScaledResidual/checkFrequency/minIter/
// minIterFactor, so any leftover GAMG key (agglomerator, cacheAgglomeration,
// mergeLevels, ...) reaches Ginkgo's config_check_decorator and aborts the process.
// Each dropped key is named in a warning - a silently ignored setting is worse than a
// slower solve.
NeoN::Dictionary makeGamgReplacement(NeoN::Dictionary& solverDict, bool distributed)
{
    // The keys that legitimately survive this entry: consumed later by
    // updatePreconditioner (smoother, nSweeps, preconditioner), by updateCriteria
    // (tolerance, relTol, maxIter) or by NeoN's own parse().
    static const std::set<std::string> keptKeys = {
        "checkFrequency",
        "configFile",
        "coupled",
        "criteria",
        "l1ScaledResidual",
        "maxIter",
        "minIter",
        "minIterFactor",
        "nSweeps",
        "preconditioner",
        "relTol",
        "reportName",
        "smoother",
        "solver",
        "tolerance",
        "type"
    };

    // OpenFOAM coarsens until a level has fewer than nCellsInCoarsestLevel cells; Ginkgo's
    // min_coarse_rows is the same knob. 64 is Ginkgo's own default.
    int minCoarseRows = 64;
    if (solverDict.contains("nCellsInCoarsestLevel"))
    {
        minCoarseRows = consumeCount(solverDict, "nCellsInCoarsestLevel");
    }

    // OpenFOAM defaults nPreSweeps to 0. A smoother with an Iteration criterion of 0
    // never runs at all, so clamp to a single sweep.
    int nSweeps = 1;
    if (solverDict.contains("nPreSweeps"))
    {
        nSweeps = std::max(1, consumeCount(solverDict, "nPreSweeps"));
    }

    // The post smoother is the pre smoother (see makeMultigridDict), so a separate
    // post-sweep count cannot be honoured - say so rather than pretend it was read.
    if (solverDict.contains("nPostSweeps"))
    {
        NeoN::Logging::warn(
            "Dropping GAMG entry nPostSweeps - the Ginkgo multigrid re-uses the pre-smoother "
            "for post-smoothing, so nPreSweeps sets both sweep counts"
        );
        solverDict.remove("nPostSweeps");
    }

    for (const auto& key : sortedKeys(solverDict))
    {
        if (keptKeys.count(key) != 0) continue;
        NeoN::Logging::warn(
            "Dropping GAMG entry {} - the Ginkgo multigrid has no equivalent setting",
            key
        );
        solverDict.remove(key);
    }

    return makeMultigridDict(minCoarseRows, nSweeps, distributed);
}

} // namespace

void updateSolver(NeoN::Dictionary& solverDict)
{
    // Map OpenFOAM solver names to NeoN/Ginkgo solver names and types
    static const std::map<std::string, std::pair<std::string, std::string>> solverMap = {
        {"PCG", {"Ginkgo", "solver::Cg"}},
        {"PBiCG", {"Ginkgo", "solver::Bicg"}},
        {"PBiCGStab", {"Ginkgo", "solver::Bicgstab"}},
        {"smoothSolver", {"Ginkgo", "solver::Bicgstab"}},
        {"GAMG", {"Ginkgo", "solver::Cg"}}, // CG-accelerated multigrid, see below
    };

    std::string& solverName = solverDict.get<std::string>("solver");
    // Copy before the assignment below overwrites it: solverName is a reference into the
    // dictionary, so testing it after the rewrite compares "Ginkgo", never the OpenFOAM name.
    const std::string foamSolverName = solverName;
    auto mapEntry = solverMap.find(foamSolverName);
    if (mapEntry != solverMap.end())
    {
ck unreachable for every solver.
        if (solverName == "GAMG")
        {
            // OpenFOAM GAMG -> Krylov-accelerated algebraic multigrid: a CG outer iteration
            // preconditioned by one V-cycle of Pgm multigrid. GAMG is selected for the SPD
            // pressure Laplacian, for which CG is the right accelerator (a bare multigrid
            // solver is markedly less robust). The outer CG stopping criteria
            // (tolerance/relTol/maxIter) are filled in afterwards by updateCriteria().
            NeoN::Logging::warn(
                "Replacing solver GAMG by solver::Cg preconditioned by Pgm algebraic multigrid"
            );
            // Assign through the reference before the dictionary is rewritten below.
            solverName = mapEntry->second.first;
            NeoN::Dictionary multigrid = makeGamgReplacement(solverDict, isDistributedRun());
            solverDict.insert("type", mapEntry->second.second);
            solverDict.insert("preconditioner", multigrid);
            return;
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

    const bool distributed = isDistributedRun();
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

    // nSweeps is how many times OpenFOAM's smoothSolver applies the smoother per outer
    // iteration. Once the smoother has become a Ginkgo preconditioner above there is no
    // equivalent knob, and Ginkgo's config parser aborts on any key it does not know -- so
    // it has to be dropped with its smoother rather than left behind.
    if (solverDict.contains("nSweeps"))
    {
        solverDict.remove("nSweeps");
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
            // OpenFOAM GAMG preconditioner -> one V-cycle of Pgm algebraic multigrid per
            // preconditioner application. Overwrites the "GAMG" string with the multigrid
            // sub-dictionary; the outer solver stays whatever fvSolution selected.
            NeoN::Logging::warn(
                "Replacing preconditioner GAMG by Pgm algebraic multigrid{}",
                distributed ? " (Schwarz-wrapped smoothers for distributed run)" : ""
            );
            solverDict.insert("preconditioner", makeGamgReplacement(solverDict, distributed));
            return;
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

    // OpenFOAM tests tolerance/relTol against the residual normalised by
    // lduMatrix::solver::normFactor. NeoN reproduces that normalisation only under its
    // L1-scaled residual criterion; without it Ginkgo compares the raw (un-normalised) L2
    // residual, so the very same dictionary values stop the solve orders of magnitude too
    // early. Default the criterion on for dictionary-mapped solvers so tolerance/relTol mean
    // the same thing on both sides; an explicit fvSolution entry still wins.
    if (!solverDict.contains("l1ScaledResidual"))
    {
        solverDict.insert("l1ScaledResidual", true);
    }

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

    // One regex key (e.g. "(U|k|epsilon)") is selected by several field names, so the
    // per-field mapping loops reach the same entry more than once. Mapping is not
    // idempotent - a second pass would reset the iteration criterion consumed from
    // maxIter - so the reportName stamp marks an entry as already mapped.
    if (solverDict.contains("reportName"))
    {
        return modSolverDict;
    }

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
    // matchKey honours OpenFOAM regex keys, e.g. "(k|omega|epsilon).*" covering both
    // `epsilon` and `epsilonFinal`.
    if (const auto finalKey = matchKey(sub, finalIter ? field + "Final" : field))
    {
        return asScalar(sub, *finalKey);
    }
    if (const auto key = matchKey(sub, field))
    {
        return asScalar(sub, *key);
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
    solverSettings(solverDict, "p") = mapFvSolution(solverSettings(solverDict, "p"));
    solverSettings(solverDict, "U") = mapFvSolution(solverSettings(solverDict, "U"));
    for (const std::string field : {"pFinal", "UFinal"})
    {
        mapSolverSettings(solverDict, field);
    }
}

} // namespace NeoFOAM
