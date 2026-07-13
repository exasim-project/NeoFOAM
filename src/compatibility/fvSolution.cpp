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
#include <cctype>
#include <fstream>
#include <iterator>
#include <sstream>

#include <NeoN/core/logging.hpp>
#include <NeoN/core/mpi/environment.hpp>
#include <NeoN/core/primitives/scalar.hpp>
#include <NeoN/core/primitives/label.hpp>
#include <NeoN/core/primitives/vec3.hpp>
#include <NeoN/core/tokenList.hpp>


namespace NeoFOAM
{

namespace
{

// Build an explicit Ginkgo Iteration stopping criterion: {type Iteration; max_iters N}.
// The multigrid solver parses its `criteria` via parse_or_get_factory_vector, which
// requires the explicit factory form (a `type` key) — the minimal {iteration N} form
// accepted by Cg/Ir's parse_or_get_criteria is NOT understood there and aborts with
// "Contains empty, but try to get string". Use the explicit form everywhere inside the
// multigrid config so it is valid for every consumer (matches Ginkgo's shipped
// pgm-multigrid-cg.json).
NeoN::Dictionary makeIterationCriterion(int maxIters)
{
    NeoN::Dictionary crit;
    crit.insert("type", std::string("Iteration"));
    crit.insert("max_iters", maxIters);
    return crit;
}

// Build a scaled-correction block-Jacobi smoother as a Ginkgo solver::Ir
// (iterative refinement / Richardson) wrapping a point-Jacobi preconditioner.
// @p nSweeps Richardson iterations are applied per smoother invocation (per
// multigrid level).
//
// scale_correction = 1 enables the experimental OpenFOAM-GAMGSolver::scale
// smoother from Ginkgo's scaleCorrectionIR branch: each sweep scales the current
// iterate by the optimal Rayleigh quotient alpha = (x·b)/(x·Ax) and adds a Jacobi
// correction D^{-1}(b - alpha*Ax). This is what makes the multigrid competitive
// with the ported SPUMA GAMG — plain Richardson (scale_correction 0) is far
// slower. Requires the patched Ginkgo pinned in NeoN (Ir::parse reads the key);
// the key would otherwise be rejected as unknown by config_check_decorator.
NeoN::Dictionary makeJacobiSmoother(int nSweeps)
{
    NeoN::Dictionary jacobi;
    jacobi.insert("type", std::string("preconditioner::Jacobi"));
    jacobi.insert("max_block_size", 1);

    NeoN::Dictionary ir;
    ir.insert("type", std::string("solver::Ir"));
    ir.insert("relaxation_factor", NeoN::scalar(0.9));
    ir.insert("solver", jacobi);
    ir.insert("scale_correction", 1);
    ir.insert("criteria", makeIterationCriterion(nSweeps));
    return ir;
}

// Build a Ginkgo algebraic-multigrid factory dictionary — the NeoN analogue of
// OpenFOAM's GAMG. Coarsening is Pgm (parallel graph match aggregation, Ginkgo's
// built-in AMG coarsening); the smoother and coarsest-level solver are damped
// block-Jacobi (see makeJacobiSmoother). Defaults are tuned for the SPD pressure
// Laplacian of cases like WindsorBody:
//   - V-cycle, up to 10 levels, coarsen until <= 64 rows (Ginkgo defaults),
//   - 1 pre/post smoother sweep (post_uses_pre defaults to true),
//   - 8 Jacobi sweeps on the coarsest level for a reasonably converged coarse solve,
//   - deterministic aggregation so the coarse hierarchy is reproducible run-to-run.
// @p nVcycles is the multigrid stopping iteration count: 1 when the multigrid is
// applied as a preconditioner (one V-cycle per application), which is how it is
// used both as a GAMG solver (CG-accelerated) and as a GAMG preconditioner.
NeoN::Dictionary makeMultigridDict(int nVcycles)
{
    NeoN::Dictionary mgLevel;
    mgLevel.insert("type", std::string("multigrid::Pgm"));
    mgLevel.insert("deterministic", true);

    NeoN::Dictionary mg;
    mg.insert("type", std::string("solver::Multigrid"));
    mg.insert("max_levels", 10);
    mg.insert("min_coarse_rows", 64);
    mg.insert("cycle", std::string("v"));
    mg.insert("mg_level", mgLevel);
    mg.insert("pre_smoother", makeJacobiSmoother(1));
    mg.insert("coarsest_solver", makeJacobiSmoother(8));
    mg.insert("default_initial_guess", std::string("zero"));
    mg.insert("criteria", makeIterationCriterion(nVcycles));
    return mg;
}

} // namespace

void updateSolver(NeoN::Dictionary& solverDict)
{
    // Map OpenFOAM solver names to NeoN/Ginkgo solver names and types
    static const std::map<std::string, std::pair<std::string, std::string>> solverMap = {
        {"PCG", {"Ginkgo", "solver::Cg"}},
        {"PBiCG", {"Ginkgo", "solver::Bicg"}},
        {"PBiCGStab", {"Ginkgo", "solver::Bicgstab"}},
        {"smoothSolver", {"Ginkgo", "solver::Ir"}},
        {"GAMG", {"Ginkgo", "solver::Multigrid"}},
    };

    std::string& solverName = solverDict.get<std::string>("solver");
    auto mapEntry = solverMap.find(solverName);
    if (mapEntry == solverMap.end())
    {
        return;
    }

    if (solverName == "GAMG")
    {
        // OpenFOAM GAMG -> Krylov-accelerated algebraic multigrid: a CG outer
        // iteration preconditioned by one V-cycle of Pgm multigrid. GAMG is
        // selected for the SPD pressure Laplacian, for which CG is the appropriate
        // accelerator (a bare multigrid solver is markedly less robust). The outer
        // CG stopping criteria (relTol/tolerance/maxIter) are filled in afterwards
        // by updateCriteria(); the multigrid's own criterion is a single V-cycle.
        NeoN::Logging::warn(
            "Mapping GAMG to Ginkgo solver::Cg preconditioned by Pgm algebraic multigrid"
        );
        solverDict.insert("type", std::string("solver::Cg"));
        solverDict.insert("preconditioner", makeMultigridDict(1));
        solverName = mapEntry->second.first; // "Ginkgo"
        return;
    }

    NeoN::Logging::warn("Replacing solver {} by {}", solverName, mapEntry->second.second);
    solverName = mapEntry->second.first;
    solverDict.insert("type", mapEntry->second.second);
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

    // A smoother (e.g. symGaussSeidel) with no explicit preconditioner maps the solver to
    // iterative refinement (solver::Ir, see solverMap), giving a stationary Jacobi smoother
    // (IR + point-Jacobi == damped Jacobi/Richardson). Default the inner operator to diagonal
    // (block-Jacobi), which is always defined and symmetric-agnostic — unlike incomplete-Cholesky
    // (DIC -> preconditioner::Ic) which is SPD-only and SIGFPEs on a non-SPD block. This block is
    // written under the "preconditioner" key; NeoN's Ginkgo parse() renames it to "solver" for
    // solver::Ir (Ginkgo names IR's inner factory "solver", which would otherwise collide with the
    // top-level "solver: Ginkgo" backend-dispatch key).
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
            // OpenFOAM GAMG preconditioner -> one V-cycle of Pgm algebraic multigrid
            // applied per preconditioner invocation. Overwrites the "GAMG" string with
            // the multigrid sub-dictionary; the outer solver is whatever fvSolution
            // selected (e.g. PCG -> Ginkgo Cg).
            NeoN::Logging::warn("Replacing preconditioner GAMG by Pgm algebraic multigrid");
            solverDict.insert("preconditioner", makeMultigridDict(1));
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

// --- Minimal brace-aware JSON scanning, used ONLY to name a configFile solver in the
// per-solve report (e.g. "config(Cg+Multigrid)"). nlohmann/json is not on this target's
// include path and we only need a couple of top-level "type" strings, so a tiny
// depth-tracking scanner is enough. Run once per field when mapFvSolution stashes
// reportName -> never on the per-solve hot path; on any malformed/missing file the caller
// falls back to the literal "configFile" label.

// Index just past the closing quote of the JSON string literal at text[i] == '"'.
std::size_t skipJsonString(const std::string& t, std::size_t i)
{
    for (++i; i < t.size(); ++i)
    {
        if (t[i] == '\\') { ++i; continue; }   // skip the escaped char
        if (t[i] == '"') return i + 1;
    }
    return i;
}

// Index of the brace/bracket that matches the one at text[open]; t.size() if unbalanced.
std::size_t matchJsonBrace(const std::string& t, std::size_t open)
{
    int depth = 0;
    for (std::size_t i = open; i < t.size(); ++i)
    {
        const char c = t[i];
        if (c == '"') { i = skipJsonString(t, i) - 1; continue; }
        if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']')
        {
            if (--depth == 0) return i;
        }
    }
    return t.size();
}

// Within the object spanning [open, close], find the IMMEDIATE-child member `key` and return
// the index of the first non-space char of its value (npos if absent). Nested members and
// string values are skipped so only top-level keys of this object match.
std::size_t findJsonMember(
    const std::string& t, std::size_t open, std::size_t close, const std::string& key
)
{
    const std::string pat = "\"" + key + "\"";
    int depth = 0;
    for (std::size_t i = open; i < close; ++i)
    {
        const char c = t[i];
        if (c == '"')
        {
            if (depth == 1 && t.compare(i, pat.size(), pat) == 0)
            {
                std::size_t j = i + pat.size();
                while (j < close && std::isspace(static_cast<unsigned char>(t[j]))) ++j;
                if (j < close && t[j] == ':')
                {
                    for (++j; j < close && std::isspace(static_cast<unsigned char>(t[j])); ++j)
                        ;
                    return j;
                }
            }
            i = skipJsonString(t, i) - 1;
            continue;
        }
        if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') --depth;
    }
    return std::string::npos;
}

// Value of the string literal starting at text[p] (p must point at '"'); empty otherwise.
std::string readJsonString(const std::string& t, std::size_t p)
{
    if (p >= t.size() || t[p] != '"') return "";
    const std::size_t end = skipJsonString(t, p);
    return t.substr(p + 1, end - p - 2);
}

// stripNamespace'd value of the immediate-child "type" of the object spanning [open, close].
std::string jsonTypeName(const std::string& t, std::size_t open, std::size_t close)
{
    const std::size_t p = findJsonMember(t, open, close, "type");
    return p == std::string::npos ? "" : stripNamespace(readJsonString(t, p));
}

// Label for one solver block: its "type", with an additive-Schwarz block unwrapped to its
// local_solver type (e.g. "Schwarz(Multigrid)") so the meaningful per-rank op is visible.
std::string jsonBlockLabel(const std::string& t, std::size_t open, std::size_t close)
{
    std::string ty = jsonTypeName(t, open, close);
    if (ty == "Schwarz")
    {
        const std::size_t lp = findJsonMember(t, open, close, "local_solver");
        if (lp != std::string::npos && lp < t.size() && t[lp] == '{')
        {
            const std::string inner = jsonTypeName(t, lp, matchJsonBrace(t, lp));
            if (!inner.empty()) ty += "(" + inner + ")";
        }
    }
    return ty;
}

// Read the configFile path out of a (mapped) solver dict. A path that parses as a single token
// is stored as a std::string, but one with '/' separators arrives as a NeoN::TokenList of the
// slash-separated components (e.g. {"system","gko","p-multigrid.json"}) -- a bare
// get<std::string>("configFile") would then throw bad_any_cast. Reconstruct the path either way,
// mirroring how the NeoN Ginkgo backend reads it (src/linearAlgebra/ginkgo/ginkgo.cpp).
std::string configFilePath(const NeoN::Dictionary& dict)
{
    const std::any& fn = dict["configFile"];
    if (fn.type() == typeid(std::string))
    {
        return std::any_cast<std::string>(fn);
    }
    auto tokens = std::any_cast<NeoN::TokenList>(fn);
    std::string path;
    for (std::size_t i = 0; i < tokens.size(); ++i)
    {
        if (i != 0) path += "/";
        path += tokens.next<std::string>();
    }
    return path;
}

// Build a report label from a Ginkgo configFile, e.g. "config(Cg+Multigrid)",
// "config(Cg+Schwarz(Multigrid))" or "config(Ir+Multigrid)". Reads the outer solver "type"
// plus its preconditioner (or, for IR/Richardson wrappers that carry the inner solver under
// "solver", that nested solver). Falls back to "configFile" on any read/parse problem.
std::string configFileLabel(const std::string& path)
{
    std::ifstream in(path);
    if (!in) return "configFile";
    const std::string t(
        (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()
    );

    const std::size_t open = t.find('{');
    if (open == std::string::npos) return "configFile";
    const std::size_t close = matchJsonBrace(t, open);

    // jsonBlockLabel (not jsonTypeName) so a top-level Schwarz solver -- the localized MG
    // promoted to the global solver -- is unwrapped to "Schwarz(Multigrid)" too.
    const std::string solver = jsonBlockLabel(t, open, close);
    if (solver.empty()) return "configFile";

    std::string secondary;
    for (const char* key : {"preconditioner", "solver"})
    {
        const std::size_t p = findJsonMember(t, open, close, key);
        if (p != std::string::npos && p < t.size() && t[p] == '{')
        {
            secondary = jsonBlockLabel(t, p, matchJsonBrace(t, p));
            break;
        }
    }

    const std::string label = secondary.empty() ? solver : solver + "+" + secondary;
    return "config(" + label + ")";
}

// Build a label from the Ginkgo solver/preconditioner ACTUALLY selected after
// mapping (e.g. "Ic+Cg", or "Schwarz(Ic)+Cg" for a distributed run). Reads the
// post-mapping dictionary, so it reflects exactly what NeoN/Ginkgo will run.
std::string ginkgoSolverLabel(const NeoN::Dictionary& mapped)
{
    if (mapped.contains("configFile")) return configFileLabel(configFilePath(mapped));

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

// Render a single std::any value held by a NeoN::Dictionary as a readable string.
// The dictionary stores values type-erased; probe the concrete types produced by the
// OpenFOAM -> NeoN conversion (word = std::string, scalar, label, Vec3) plus the
// int/bool/float literals the mapping code inserts. Unknown types degrade to their
// demangled type name instead of throwing, so logging the config never aborts a run.
std::string anyToString(const std::any& value)
{
    if (const auto* p = std::any_cast<std::string>(&value)) return *p;
    if (const auto* p = std::any_cast<NeoN::scalar>(&value)) return std::to_string(*p);
    if (const auto* p = std::any_cast<NeoN::label>(&value)) return std::to_string(*p);
    if (const auto* p = std::any_cast<int>(&value)) return std::to_string(*p);
    if (const auto* p = std::any_cast<bool>(&value)) return *p ? "true" : "false";
    if (const auto* p = std::any_cast<float>(&value)) return std::to_string(*p);
    if (const auto* p = std::any_cast<double>(&value)) return std::to_string(*p);
    if (const auto* p = std::any_cast<NeoN::Vec3>(&value))
    {
        std::ostringstream os;
        os << *p;
        return os.str();
    }
    return "<" + NeoN::demangle(value.type().name()) + ">";
}

// Pretty-print a NeoN::Dictionary as an indented block (sub-dictionaries nested
// recursively). Keys are sorted so the output is stable across the unordered_map's
// arbitrary iteration order, which makes successive runs / log diffs comparable.
std::string dictToString(const NeoN::Dictionary& dict, int indent = 0)
{
    const std::string pad(static_cast<std::size_t>(indent) * 4, ' ');
    std::string out = "{\n";
    std::vector<std::string> keys = dict.keys();
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys)
    {
        out += pad + "    " + key + ": ";
        if (dict.isDict(key))
        {
            out += dictToString(dict.subDict(key), indent + 1);
        }
        else
        {
            out += anyToString(dict.getMap().at(key));
        }
        out += "\n";
    }
    out += pad + "}";
    return out;
}

// Effective `optimize` flag (DSL expression optimization). Mirrors the default/parse
// rule of the PDE assembly path (pde.hpp): an absent key reads as "false". OpenFOAM
// tokenizes the switch (`optimize true;`) as a word, hence the std::string lookup.
std::string effectiveOptimize(const NeoN::Dictionary& dict)
{
    return dict.isType<std::string>("optimize") ? dict.get<std::string>("optimize") : "false";
}

// Effective `checkFrequency` (how often the L1 criterion evaluates the true residual),
// using the same default the Ginkgo control seeds (ginkgo.hpp): 1. OpenFOAM stores it
// as a label after conversion; tolerate int/scalar too, matching readInt's coercion.
long effectiveCheckFrequency(const NeoN::Dictionary& dict)
{
    if (dict.isType<int>("checkFrequency")) return dict.get<int>("checkFrequency");
    if (dict.isType<NeoN::label>("checkFrequency"))
        return static_cast<long>(dict.get<NeoN::label>("checkFrequency"));
    if (dict.isType<NeoN::scalar>("checkFrequency"))
        return static_cast<long>(dict.get<NeoN::scalar>("checkFrequency"));
    return 1;
}

// Emit the resolved solver configuration NeoN/Ginkgo will run for one field, so the
// OpenFOAM-derived settings are visible in the log. Prints the full (post-mapping)
// dictionary plus the effective values of the easy-to-miss tuning keys `optimize`
// and `checkFrequency`, which are shown even when defaulted.
void logSolverConfig(const NeoN::Dictionary& mapped, const std::string& origin)
{
    NeoN::Logging::info(
        "NeoFOAM solver config ({}) for '{}':\n{}\n    effective optimize: {}\n    effective "
        "checkFrequency: {}",
        origin,
        mapped.isType<std::string>("reportName") ? mapped.get<std::string>("reportName")
                                                 : std::string("?"),
        dictToString(mapped),
        effectiveOptimize(mapped),
        effectiveCheckFrequency(mapped)
    );
}

} // namespace

NeoN::Dictionary mapFvSolution(const NeoN::Dictionary& solverDict)
{
    NeoN::Dictionary modSolverDict = solverDict;

    if (solverDict.contains("configFile"))
    {
        // SolverFactory::create() dispatches on `solver`; the configFile path
        // doesn't write one (the user only specified configFile + tolerances)
        // so the lookup throws `Key not found: solver`. Inject "Ginkgo" so
        // the factory routes to GinkgoSolver, which itself reads configFile.
        if (!modSolverDict.contains("solver"))
        {
            modSolverDict.insert("solver", std::string("Ginkgo"));
        }
        // Name the configFile solver stack in the per-solve report by reading the JSON
        // (e.g. "config(Cg+Multigrid)") instead of the opaque literal "configFile".
        modSolverDict.insert(
            "reportName", configFileLabel(configFilePath(solverDict))
        );
        logSolverConfig(modSolverDict, "from fvSolution configFile");
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

    // Echo the fully-mapped solver settings NeoN/Ginkgo will run, so the
    // OpenFOAM-derived config is visible in the log (this is the post-mapping
    // dict -> reflects exactly what the linear solver receives).
    logSolverConfig(modSolverDict, "mapped from fvSolution");

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
