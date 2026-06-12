// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
/* This file implements comparison operator to compare OpenFOAM and corresponding NeoFOAM fields
 * TODO the comparison operator only make sense for testing purposes
 * so this should be part of the tests
 */


#include "NeoFOAM/compatibility/fvSolution.hpp"

#include <map>
#include <optional>

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

namespace
{

// Return a pointer to the ParIc/ParIlu factorization sub-dictionary inside a mapped
// preconditioner dict, or nullptr if the preconditioner has no factorization (e.g. Jacobi).
// Handles both the direct form (preconditioner::Ic -> "factorization") and the distributed
// additive-Schwarz wrapper (preconditioner::Schwarz -> "local_solver" -> "factorization").
NeoN::Dictionary* factorizationSubDict(NeoN::Dictionary& precond)
{
    if (precond.isDict("factorization"))
    {
        return &precond.subDict("factorization");
    }
    if (precond.isDict("local_solver"))
    {
        NeoN::Dictionary& local = precond.subDict("local_solver");
        if (local.isDict("factorization"))
        {
            return &local.subDict("factorization");
        }
    }
    return nullptr;
}

// Return a pointer to the factorization-based preconditioner sub-dictionary itself
// (the one carrying `type: preconditioner::Ic` / `preconditioner::Ilu`), i.e. the dict
// where an `l_solver` / `u_solver` apply override belongs as a sibling of `factorization`.
// Unlike factorizationSubDict, this returns the preconditioner dict, not the nested
// `factorization`. Unwraps the distributed additive-Schwarz wrapper (Schwarz ->
// local_solver). Returns nullptr for preconditioners without a factorization (e.g. Jacobi).
NeoN::Dictionary* factorizationPrecondDict(NeoN::Dictionary& precond)
{
    if (precond.isDict("factorization"))
    {
        return &precond;
    }
    if (precond.isDict("local_solver"))
    {
        NeoN::Dictionary& local = precond.subDict("local_solver");
        if (local.isDict("factorization"))
        {
            return &local;
        }
    }
    return nullptr;
}

// Build a Ginkgo ISAI (Incomplete Sparse Approximate Inverse) factory config for use as the
// apply solver of an incomplete factorization. `isaiType` is "lower" or "upper". ISAI
// approximates the triangular factor's inverse as a sparse matrix, so the apply becomes a
// single parallel sparse mat-vec instead of Ginkgo's default exact sparse triangular solve
// (solver::LowerTrs), which serializes along dependency chains and rebuilds an analysis phase
// every solve -- the dominant cost of Ic/Ilu on GPUs. `sparsityPower` (Ginkgo default 1)
// trades approximation quality for cost: higher powers use a denser inverse pattern.
NeoN::Dictionary
makeIsaiSolverDict(const std::string& isaiType, const std::optional<int>& sparsityPower)
{
    NeoN::Dictionary isai(
        {{std::string("type"), std::string("preconditioner::Isai")},
         {std::string("isai_type"), isaiType}}
    );
    if (sparsityPower)
    {
        isai.insert("sparsity_power", *sparsityPower);
    }
    return isai;
}

// Build a Ginkgo Ir (iterative refinement / Richardson) factory config for use as the apply
// solver of an incomplete factorization. With a Jacobi inner solver and a FIXED sweep count this
// performs `sweeps` damped-Jacobi sweeps on the triangular factor -- a parallel, setup-free
// approximate triangular solve, the direct analogue of SPUMA's aDIC (which uses a single sweep).
// The sweep count MUST be a fixed iteration criterion, never a residual one: a residual-stopped
// inner solve makes the preconditioner non-linear/variable across applies, which breaks the outer
// CG (Krylov assumes a constant preconditioner). `relaxationFactor` is the Richardson damping
// omega (Ginkgo default 1.0).
//
// Note the criterion must use Ginkgo's EXPLICIT factory form `{type: Iteration, max_iters: N}`.
// Ir parses `criteria` via parse_or_get_factory_vector, which treats a map as a single criterion
// factory keyed on `type`; the bare `{iteration: N}` shorthand is only understood by the
// top-level solver's parse_minimal_criteria path, not here.
NeoN::Dictionary makeIrSolverDict(int sweeps, const std::optional<NeoN::scalar>& relaxationFactor)
{
    NeoN::Dictionary criteria(
        {{std::string("type"), std::string("Iteration")}, {std::string("max_iters"), sweeps}}
    );
    NeoN::Dictionary inner(
        {{std::string("type"), std::string("preconditioner::Jacobi")},
         {std::string("max_block_size"), 1}}
    );
    NeoN::Dictionary ir(
        {{std::string("type"), std::string("solver::Ir")},
         {std::string("criteria"), criteria},
         {std::string("solver"), inner}}
    );
    if (relaxationFactor)
    {
        ir.insert("relaxation_factor", *relaxationFactor);
    }
    return ir;
}

} // namespace

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

    // Optional sweep count for the ParIc/ParIlu factorization (mapped to Ginkgo's `iterations`).
    // OpenFOAM-style top-level key, e.g. `nSweeps 3;`. The number of fixed-point sweeps the
    // factorization needs grows with parallelism: ~3 on CPU, ~10 on GPU (per Ginkgo's own
    // guidance). Read and strip it up front so it never leaks into the Ginkgo solver config,
    // whichever preconditioner branch runs below. Integer tokens from an OpenFOAM dictionary may
    // arrive as int or label.
    std::optional<int> nSweeps;
    if (solverDict.contains("nSweeps"))
    {
        if (solverDict.isType<int>("nSweeps"))
        {
            nSweeps = solverDict.get<int>("nSweeps");
        }
        else if (solverDict.isType<NeoN::label>("nSweeps"))
        {
            nSweeps = static_cast<int>(solverDict.get<NeoN::label>("nSweeps"));
        }
        else
        {
            nSweeps = static_cast<int>(solverDict.get<NeoN::scalar>("nSweeps"));
        }
        solverDict.remove("nSweeps");
    }

    // Optional apply-solver override for the factorization preconditioners (DIC -> Ic,
    // DILU -> Ilu). By default Ginkgo applies the incomplete factor with an EXACT sparse
    // triangular solve (solver::LowerTrs), whose dependency chains serialize on GPUs and
    // whose analysis phase is rebuilt every solve -- the reason Ic/Ilu lag a plain diagonal
    // (Jacobi) preconditioner on the device. `lSolver isai;` swaps that for an ISAI
    // approximate-inverse apply (a parallel sparse mat-vec) and `lSolver ir;` for a fixed number
    // of Jacobi sweeps (Ginkgo Ir) -- both GPU-friendly analogues of SPUMA's aDIC. Read and strip
    // up front so the key never leaks into the Ginkgo config. Recognised values: "trs"/"exact"
    // (default, no-op), "isai", and "ir".
    std::optional<std::string> lSolver;
    if (solverDict.contains("lSolver"))
    {
        lSolver = solverDict.get<std::string>("lSolver");
        solverDict.remove("lSolver");
    }

    // Optional ISAI sparsity power (Ginkgo default 1); higher = denser inverse pattern,
    // better approximation at higher cost. Only meaningful with `lSolver isai;`.
    std::optional<int> sparsityPower;
    if (solverDict.contains("sparsityPower"))
    {
        if (solverDict.isType<int>("sparsityPower"))
        {
            sparsityPower = solverDict.get<int>("sparsityPower");
        }
        else if (solverDict.isType<NeoN::label>("sparsityPower"))
        {
            sparsityPower = static_cast<int>(solverDict.get<NeoN::label>("sparsityPower"));
        }
        else
        {
            sparsityPower = static_cast<int>(solverDict.get<NeoN::scalar>("sparsityPower"));
        }
        solverDict.remove("sparsityPower");
    }

    // Optional Ir (approximate triangular solve) controls; only meaningful with `lSolver ir;`.
    // lSolverSweeps: number of fixed Jacobi sweeps (Ir iteration criterion); default 1, the
    // single-sweep aDIC analogue. Integer tokens may arrive as int or label.
    std::optional<int> lSolverSweeps;
    if (solverDict.contains("lSolverSweeps"))
    {
        if (solverDict.isType<int>("lSolverSweeps"))
        {
            lSolverSweeps = solverDict.get<int>("lSolverSweeps");
        }
        else if (solverDict.isType<NeoN::label>("lSolverSweeps"))
        {
            lSolverSweeps = static_cast<int>(solverDict.get<NeoN::label>("lSolverSweeps"));
        }
        else
        {
            lSolverSweeps = static_cast<int>(solverDict.get<NeoN::scalar>("lSolverSweeps"));
        }
        solverDict.remove("lSolverSweeps");
    }

    // relaxationFactor: Richardson damping omega for the Ir sweeps (Ginkgo default 1.0).
    std::optional<NeoN::scalar> relaxationFactor;
    if (solverDict.contains("relaxationFactor"))
    {
        relaxationFactor = solverDict.isType<int>("relaxationFactor")
                             ? NeoN::scalar(solverDict.get<int>("relaxationFactor"))
                             : solverDict.get<NeoN::scalar>("relaxationFactor");
        solverDict.remove("relaxationFactor");
    }

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
            NeoN::Dictionary precond = mapEntry->second; // mutable copy to inject sweep count

            // Factorization preconditioners (DIC -> Ic/ParIc, DILU -> Ilu/ParIlu) need an
            // incomplete factorization of the system matrix. Two adjustments:
            //  1. Forward the optional sweep count to the factorization's `iterations`.
            //  2. The OpenFOAM pressure Laplacian is assembled negative-(semi)definite, but
            //     incomplete Cholesky requires a positive-definite matrix. Request the solver
            //     negate the system (solve (-A) x = (-b), same x and residual) so Ginkgo is
            //     handed an SPD matrix. Jacobi (diagonal) has no factorization and is untouched.
            if (NeoN::Dictionary* fac = factorizationSubDict(precond))
            {
                if (nSweeps)
                {
                    fac->insert("iterations", *nSweeps);
                }
                solverDict.insert("negateSystem", true);
            }

            // Optional approximate apply override (ISAI or Ir). Only valid for the
            // factorization preconditioners (Ic/Ilu); fail loud on a misapplied or
            // misspelled key so a benchmark run can't silently fall back to the slow
            // exact-triangular-solve default.
            if (lSolver)
            {
                const std::string& mode = *lSolver;
                if (mode == "isai" || mode == "ir")
                {
                    NeoN::Dictionary* facPrecond = factorizationPrecondDict(precond);
                    if (facPrecond == nullptr)
                    {
                        throw std::runtime_error(
                            "\n'lSolver " + mode
                            + ";' applies only to factorization preconditioners (DIC -> Ic, "
                              "DILU -> Ilu); preconditioner '"
                            + preconditionerName + "' has no triangular factor to approximate.\n"
                        );
                    }
                    // Ic applies the apply solver to L and its conjugate-transpose to L^H, so
                    // only a lower override is needed. Ilu is asymmetric and needs both L and U.
                    const bool isIlu =
                        facPrecond->get<std::string>("type") == "preconditioner::Ilu";
                    if (mode == "isai")
                    {
                        facPrecond->insert("l_solver", makeIsaiSolverDict("lower", sparsityPower));
                        if (isIlu)
                        {
                            facPrecond->insert(
                                "u_solver", makeIsaiSolverDict("upper", sparsityPower)
                            );
                        }
                    }
                    else // "ir": a fixed number of Jacobi sweeps (default 1 == aDIC)
                    {
                        const int sweeps = lSolverSweeps.value_or(1);
                        facPrecond->insert("l_solver", makeIrSolverDict(sweeps, relaxationFactor));
                        if (isIlu)
                        {
                            facPrecond->insert(
                                "u_solver", makeIrSolverDict(sweeps, relaxationFactor)
                            );
                        }
                    }
                }
                else if (mode != "trs" && mode != "exact")
                {
                    throw std::runtime_error(
                        "\nUnknown lSolver '" + mode
                        + "'. Valid values: 'isai' (approximate-inverse apply), 'ir' (fixed "
                          "Jacobi-sweep approximate triangular solve), or 'trs'/'exact' (default "
                          "exact triangular solve).\n"
                    );
                }
            }

            NeoN::Logging::warn(
                "Replacing preconditioner {} by {}{}",
                preconditionerName,
                precond.get<std::string>("type"),
                distributed ? " (Schwarz-wrapped for distributed run)" : ""
            );
            solverDict.insert("preconditioner", precond);
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
                // Annotate a factorization preconditioner with its apply-solver override
                // (e.g. "Ic(Isai)") so an ISAI run is distinguishable from the default
                // exact-triangular-solve run in benchmark logs.
                auto applySuffix = [](const NeoN::Dictionary& p) -> std::string
                {
                    if (p.isDict("l_solver") && p.subDict("l_solver").contains("type"))
                    {
                        return "(" + stripNamespace(p.subDict("l_solver").get<std::string>("type"))
                             + ")";
                    }
                    return "";
                };
                // Unwrap the additive-Schwarz local solver so the meaningful
                // per-rank preconditioner is visible in distributed runs.
                if (ptype == "preconditioner::Schwarz" && pd.contains("local_solver")
                    && pd.isDict("local_solver"))
                {
                    const NeoN::Dictionary& ld = pd.subDict("local_solver");
                    if (ld.contains("type"))
                    {
                        precond += "(" + stripNamespace(ld.get<std::string>("type"))
                                 + applySuffix(ld) + ")";
                    }
                }
                else
                {
                    precond += applySuffix(pd);
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

} // namespace NeoFOAM
