#!/usr/bin/env python3
"""Generate Ginkgo multigrid-preconditioner variants for the occDrivAre p-solver
parameter study, from the production p-multigrid.json base.

Grid:
    max_levels        in {2, 4, 10, 15}
    scale_correction  in {0}   (applied to BOTH pre_smoother and post_smoother;
                                the scale_correction=2 cases were removed from the sweep)

scale_correction is the solver::Ir scale_correction_mode enum. In the updated ginkgo
(NeoN_GINKGO_TAG 241deca) it must be the enum STRING -- "none" | "forward" | "backward" --
NOT an integer; a numeric value makes Ir::parse call pnode::get_string() on an int pnode and
std::abort during config parse. The grid is kept in the historical integer ordinal (0=none,
1=forward, 2=backward) for the sc<n> filename, and mapped to the enum string via SC_MODE on
write. coarsest_solver.scale_correction is left untouched (it carries p-multigrid.json's value).
Output files:
    p-multigrid.L<levels>.sc<scale>.json

Outer stopping criterion (inherited by every generated variant from p-multigrid.json):
NeoN's OpenFOAM-equivalent L1-scaled residual criterion, referenced by name
("neon::l1ScaledResidual") in the criteria array, plus a 150-iteration cap. NeoN registers
this criterion into the Ginkgo config registry (see la::ginkgo::makeL1CriterionFactory), so
it stops on sum|b - A x| / normFactor with the OpenFOAM L1 normFactor -- the true L1-scaled
residual, not the L2 rhs_norm analog. The absolute tolerance (1e-7) and relTol come from the
fvSolution p-block, which must opt in with `l1ScaledResidual true;` (and tolerance/relTol);
see fvSolution.case3.
"""
import json
import os
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "p-multigrid.json")

MAX_LEVELS = [2, 4, 10, 15]
SCALE_CORRECTION = [0]  # scale_correction=2 cases removed from the sweep
# Ir scale_correction_mode: integer ordinal (used for the sc<n> filename) -> ginkgo enum
# string (required by the config parser; an int aborts in Ir::parse via pnode::get_string()).
SC_MODE = {0: "none", 1: "forward", 2: "backward"}

with open(BASE) as f:
    base = json.load(f)

written = []
for levels, scale in product(MAX_LEVELS, SCALE_CORRECTION):
    cfg = json.loads(json.dumps(base))  # deep copy
    pre = cfg["preconditioner"]
    pre["max_levels"] = levels
    pre["pre_smoother"][0]["scale_correction"] = SC_MODE[scale]
    pre["post_smoother"][0]["scale_correction"] = SC_MODE[scale]

    name = f"p-multigrid.L{levels}.sc{scale}.json"
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    written.append(name)

# Derived solver-type variants for the param-study-mg.sh fcg / mgsolver runs:
#   p-fcg-multigrid.json    FCG outer solver + the same Multigrid preconditioner.
#   p-multigrid-solver.json Multigrid promoted to the GLOBAL solver (no outer Krylov):
#                           the preconditioner block carrying the base's outer
#                           convergence criteria instead of the single-cycle criteria.
fcg = json.loads(json.dumps(base))
fcg["type"] = "solver::Fcg"
with open(os.path.join(HERE, "p-fcg-multigrid.json"), "w") as f:
    json.dump(fcg, f, indent=4)
    f.write("\n")
written.append("p-fcg-multigrid.json")

mgSolver = json.loads(json.dumps(base["preconditioner"]))
mgSolver["criteria"] = json.loads(json.dumps(base["criteria"]))
with open(os.path.join(HERE, "p-multigrid-solver.json"), "w") as f:
    json.dump(mgSolver, f, indent=4)
    f.write("\n")
written.append("p-multigrid-solver.json")

# p-multigrid-directcoarse.json: base Cg+Multigrid but the coarsest level is solved
# EXACTLY by a direct LU solver instead of the iterative Ir/Schwarz coarse solver.
# Lu (general) is used rather than Cholesky so a non-symmetric coarse Galerkin
# operator is still handled.
direct = json.loads(json.dumps(base))
direct["preconditioner"]["coarsest_solver"] = {
    "type": "solver::Direct",
    "factorization": {"type": "factorization::Lu"},
}
with open(os.path.join(HERE, "p-multigrid-directcoarse.json"), "w") as f:
    json.dump(direct, f, indent=4)
    f.write("\n")
written.append("p-multigrid-directcoarse.json")

# p-multigrid-smooth2.json: base Cg+Multigrid but the V-cycle uses a stronger smoother
# of 2 Ir/Jacobi iterations (max_iters=2) on both pre_smoother and post_smoother instead
# of the base's weak single sweep. Tests whether more smoothing per level reduces the
# outer CG iteration count.
smooth2 = json.loads(json.dumps(base))
for s in ("pre_smoother", "post_smoother"):
    for crit in smooth2["preconditioner"][s][0]["criteria"]:
        if crit.get("type") == "Iteration":
            crit["max_iters"] = 2
with open(os.path.join(HERE, "p-multigrid-smooth2.json"), "w") as f:
    json.dump(smooth2, f, indent=4)
    f.write("\n")
written.append("p-multigrid-smooth2.json")

# p-multigrid-cgcoarse.json: base Cg+Multigrid but the coarsest level is solved by an
# inner Jacobi-preconditioned CG (Krylov) instead of the base's fixed Ir/Schwarz sweeps.
# Tighter than the 8 Ir iterations and self-adapting, while staying matrix-free unlike
# the direct (LU) coarse solver. The Jacobi preconditioner MUST be wrapped in a Schwarz
# (per-rank local solver): in the distributed (-parallel) run a bare Jacobi can't extract
# a diagonal from the distributed coarse operator and aborts in Jacobi::generate. This
# mirrors the Schwarz{Jacobi} wrapping the base uses in every smoother/coarsest block.
cgcoarse = json.loads(json.dumps(base))
cgcoarse["preconditioner"]["coarsest_solver"] = {
    "type": "solver::Cg",
    "preconditioner": {
        "type": "preconditioner::Schwarz",
        "local_solver": {
            "type": "preconditioner::Jacobi",
            "max_block_size": 1,
        },
    },
    "criteria": [
        {"type": "Iteration", "max_iters": 30},
        {
            "type": "ResidualNorm",
            "reduction_factor": 1e-4,
            "baseline": "initial_resnorm",
        },
    ],
}
with open(os.path.join(HERE, "p-multigrid-cgcoarse.json"), "w") as f:
    json.dump(cgcoarse, f, indent=4)
    f.write("\n")
written.append("p-multigrid-cgcoarse.json")

# p-multigrid-localized.json: LOCALIZED multigrid preconditioner (+ localized smoother).
# The base is a DISTRIBUTED multigrid -- Multigrid is generated on the full distributed
# matrix, distributed Pgm coarsens it, and every smoother is a distributed Schwarz{Jacobi}.
# This variant instead generates the WHOLE Multigrid on each rank's LOCAL block and couples
# the ranks with a single outer restricted-additive Schwarz:
#     solver::Cg  ->  preconditioner::Schwarz { local_solver: solver::Multigrid {...} }
# Because the inner Multigrid now acts on a non-distributed local operator, its smoothers
# and coarsest solver drop the per-smoother Schwarz wrap and use a plain LOCAL Jacobi
# (the "localized smoother"). This is the config-file equivalent of the OGL type_=="Schwarz"
# construction (MG on get_local() wrapped by wrap_schwarz), as opposed to the base's
# type_=="Distributed" path. Compared with the distributed base it trades inter-rank
# coarse-grid coupling for a cheaper, embarrassingly-parallel per-rank V-cycle, so expect
# more outer CG iterations but cheaper per-cycle work.
def _localize_smoother(block):
    # Unwrap Schwarz{local_solver: X} -> X so the smoother runs as a plain local solver.
    solver = block.get("solver")
    if isinstance(solver, dict) and solver.get("type") == "preconditioner::Schwarz":
        block["solver"] = solver["local_solver"]

localized_mg = json.loads(json.dumps(base["preconditioner"]))
for s in ("pre_smoother", "post_smoother"):
    for blk in localized_mg[s]:
        _localize_smoother(blk)
_localize_smoother(localized_mg["coarsest_solver"])
localized = {
    "type": base["type"],
    "preconditioner": {
        "type": "preconditioner::Schwarz",
        "local_solver": localized_mg,
    },
    "criteria": json.loads(json.dumps(base["criteria"])),
}
with open(os.path.join(HERE, "p-multigrid-localized.json"), "w") as f:
    json.dump(localized, f, indent=4)
    f.write("\n")
written.append("p-multigrid-localized.json")

# p-multigrid-localized.L<levels>.json: the LOCALIZED multigrid (p-multigrid-localized.json)
# swept over the inner Multigrid's max_levels. The per-rank Schwarz{Multigrid(local)}
# construction and its localized local-Jacobi smoothers are unchanged; only max_levels varies.
# Drives the localized max_levels study in param-study-mg-tuning.sh (run name
# pMG-localized-L<levels>-ukoSmooth).
LOCALIZED_LEVELS = [2, 4, 6, 8, 10, 15, 20]
for levels in LOCALIZED_LEVELS:
    cfg = json.loads(json.dumps(localized))
    cfg["preconditioner"]["local_solver"]["max_levels"] = levels
    name = f"p-multigrid-localized.L{levels}.json"
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    written.append(name)

# p-multigrid-localized-mp.json: the LOCALIZED multigrid (p-multigrid-localized.json) with its
# OUTER stopping criterion swapped from NeoN's L1-scaled residual ("neon::l1ScaledResidual") to a
# plain Ginkgo RELATIVE ResidualNorm (reduction_factor 0.01 == the case3 relTol, baseline
# initial_resnorm). REQUIRED for the mixed-precision study (param-study-mp.sh): NeoN's GinkgoSolver
# refuses innerPrecision combined with an l1ScaledResidual-in-config (the l1 criterion is fp64-typed
# and cannot be embedded in the lower-precision inner factory -- see ginkgo.hpp NF_ERROR_EXIT), and
# the named criterion would be UNREGISTERED once the fvSolution p-block drops `l1ScaledResidual true`
# (it is only emplaced into the config registry when l1 control is read). The preconditioner /
# smoother construction is otherwise byte-identical to p-multigrid-localized.json, so the only
# variable the mp study changes against the localized base is the inner solve precision.
localized_mp = json.loads(json.dumps(localized))
localized_mp["criteria"] = [
    {"type": "Iteration", "max_iters": 150},
    {"type": "ResidualNorm", "reduction_factor": 0.01, "baseline": "initial_resnorm"},
]
with open(os.path.join(HERE, "p-multigrid-localized-mp.json"), "w") as f:
    json.dump(localized_mp, f, indent=4)
    f.write("\n")
written.append("p-multigrid-localized-mp.json")

# p-multigrid-localized-precfloat.json / -precbf16.json: the LOCALIZED multigrid with ONLY the
# PRECONDITIONER subtree built in reduced precision, via the Ginkgo config per-node "value_type"
# override placed on the preconditioner (Schwarz{Multigrid(local)}) node. The OUTER solver::Cg, its
# Krylov vectors, and the l1ScaledResidual stop all stay fp64; Ginkgo converts the residual to the
# lower precision at the preconditioner-apply boundary (Schwarz::apply ->
# precision_dispatch_real_complex_distributed) and back, so this is a genuinely MIXED-precision
# preconditioner. This is DISTINCT from the param-study-mp.sh innerPrecision runs, which instead
# drop the WHOLE inner Cg+MG solve to reduced precision under an outer fp64 Ir. Because the outer Cg
# stays fp64, the l1 criterion remains valid and is KEPT -- so these run through the normal
# configFile path with `l1ScaledResidual true`, exactly like the cache-rebuild reference (no
# innerPrecision key). value_type strings are Ginkgo's config type names (float32, bfloat16).
for value_type, suffix in (("float32", "precfloat"), ("bfloat16", "precbf16")):
    cfg = json.loads(json.dumps(localized))
    cfg["preconditioner"]["value_type"] = value_type
    name = f"p-multigrid-localized-{suffix}.json"
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    written.append(name)

# p-multigrid-localized-solver.json: the same LOCALIZED multigrid promoted to the GLOBAL
# solver (no outer Krylov) -- the outer Schwarz{Multigrid(local)} block carrying the base's
# outer convergence criteria instead of being a Cg preconditioner. Counterpart of
# p-multigrid-solver.json for the localized construction.
localizedSolver = json.loads(json.dumps(localized["preconditioner"]))
localizedSolver["criteria"] = json.loads(json.dumps(base["criteria"]))
with open(os.path.join(HERE, "p-multigrid-localized-solver.json"), "w") as f:
    json.dump(localizedSolver, f, indent=4)
    f.write("\n")
written.append("p-multigrid-localized-solver.json")

# p-multigrid-scalecorr.json: SCALE-CORRECTED multigrid (new NeoN_GINKGO_TAG feature; the
# scale_correction config keys ship in tag 241deca -- the productionnvidia_h200 build. The
# profilingnvidia_h200 binary used by the study is on the older ginkgo and would ABORT on the
# MG-level "scale_correction" key, so this variant needs a rebuild at NeoN_GINKGO_TAG.)
#
# Mirrors Ginkgo's scale-correction example: an OUTER solver::Ir carrying
# scale_correction="backward" (the backward Rayleigh correction that pre-conditions the
# residual before handing it to the V-cycle as an initial guess) wraps a solver::Multigrid
# that itself enables per-level scale_correction=true. The MG is pinned to exactly one V-cycle
# per outer IR iteration (criteria: Iteration max_iters=1), so one outer IR step == one
# V-cycle. Config-key types (tag 241deca): solver::Multigrid scale_correction is a BOOL;
# solver::Ir scale_correction is the scale_correction_mode enum string none|forward|backward.
#
# Differences from the serial Ginkgo example: the smoothers keep the distributed
# Schwarz{Jacobi} wrap (a bare Jacobi can't extract a diagonal from the distributed operator
# and aborts in Jacobi::generate), and the outer convergence criteria use the study's L1
# criterion (neon::l1ScaledResidual + a 150-iteration cap) instead of the example's
# rhs_norm ResidualNorm, so it stops on the same residual measure as every other run.
schwarz_jacobi = {
    "type": "preconditioner::Schwarz",
    "local_solver": {"type": "preconditioner::Jacobi", "max_block_size": 1},
}
scalecorr = {
    "type": "solver::Ir",
    "scale_correction": "backward",
    "solver": {
        "type": "solver::Multigrid",
        "max_levels": 10,
        "min_coarse_rows": 64,
        "cycle": "v",
        "default_initial_guess": "zero",
        "post_uses_pre": True,
        "scale_correction": True,
        "mg_level": [{"type": "multigrid::Pgm", "deterministic": True}],
        "pre_smoother": [
            {
                "type": "solver::Ir",
                "relaxation_factor": 0.9,
                "solver": json.loads(json.dumps(schwarz_jacobi)),
                "criteria": [{"type": "Iteration", "max_iters": 2}],
            }
        ],
        "coarsest_solver": {
            "type": "solver::Ir",
            "relaxation_factor": 0.9,
            "solver": json.loads(json.dumps(schwarz_jacobi)),
            "criteria": [{"type": "Iteration", "max_iters": 4}],
        },
        "criteria": [{"type": "Iteration", "max_iters": 1}],
    },
    "criteria": json.loads(json.dumps(base["criteria"])),
}
with open(os.path.join(HERE, "p-multigrid-scalecorr.json"), "w") as f:
    json.dump(scalecorr, f, indent=4)
    f.write("\n")
written.append("p-multigrid-scalecorr.json")

# p-multigrid-scalecorr-localized.json: the SCALE-CORRECTED multigrid made LOCALIZED -- the
# combination of p-multigrid-scalecorr.json and p-multigrid-localized.json. The outer
# solver::Ir(scale_correction="backward") and the inner solver::Multigrid(scale_correction=true)
# are kept exactly as in scalecorr (so the Rayleigh backward correction still pre-conditions the
# residual before each V-cycle), but the inner Multigrid is now generated on each rank's LOCAL
# block and coupled by a single outer restricted-additive Schwarz:
#     solver::Ir[backward] -> preconditioner::Schwarz { local_solver: solver::Multigrid[scale_corr] }
# Because the inner Multigrid acts on a non-distributed local operator, its smoothers and
# coarsest solver drop the per-smoother Schwarz wrap for a plain LOCAL Jacobi (the localized
# smoother), exactly as in p-multigrid-localized.json. Combines the scalecorr variant's scale
# correction with the localized variant's cheaper, embarrassingly-parallel per-rank V-cycle.
# Needs the scale_correction config keys from NeoN_GINKGO_TAG (241deca) -- the production build.
scalecorr_localized = json.loads(json.dumps(scalecorr))
_inner_mg = scalecorr_localized["solver"]
for blk in _inner_mg["pre_smoother"]:
    _localize_smoother(blk)
_localize_smoother(_inner_mg["coarsest_solver"])
scalecorr_localized["solver"] = {
    "type": "preconditioner::Schwarz",
    "local_solver": _inner_mg,
}
with open(os.path.join(HERE, "p-multigrid-scalecorr-localized.json"), "w") as f:
    json.dump(scalecorr_localized, f, indent=4)
    f.write("\n")
written.append("p-multigrid-scalecorr-localized.json")

print(f"wrote {len(written)} variants:")
for n in written:
    print(" ", n)
