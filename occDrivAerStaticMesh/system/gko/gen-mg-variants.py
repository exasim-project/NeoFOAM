#!/usr/bin/env python3
"""Generate Ginkgo multigrid-preconditioner variants for the occDrivAer p-solver
parameter study, from the production p-multigrid.json base.

Grid:
    max_levels        in {2, 4, 10, 15}
    scale_correction  in {0, 2}   (applied to BOTH pre_smoother and post_smoother)

coarsest_solver.scale_correction is left untouched. Output files:
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
SCALE_CORRECTION = [0, 2]

with open(BASE) as f:
    base = json.load(f)

written = []
for levels, scale in product(MAX_LEVELS, SCALE_CORRECTION):
    cfg = json.loads(json.dumps(base))  # deep copy
    pre = cfg["preconditioner"]
    pre["max_levels"] = levels
    pre["pre_smoother"][0]["scale_correction"] = scale
    pre["post_smoother"][0]["scale_correction"] = scale

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

print(f"wrote {len(written)} variants:")
for n in written:
    print(" ", n)
