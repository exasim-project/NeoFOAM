#!/bin/bash
#
# COARSE-GRID-SOLVER study for occDrivAerStaticMesh, on the new best-practice precfloat setup
# (FLOAT MG preconditioner).
#
# Base config: system/gko/p-multigrid-localized-precfloat.json -- the localized
# Schwarz{Multigrid(local)} preconditioner with "value_type":"float32" (the validated float MG
# preconditioner; outer solver::Cg and the l1ScaledResidual stop stay fp64). Everything is held
# fixed EXCEPT the multigrid coarsest_solver, which is swept over:
#
#   solver:  CG  (solver::Cg + local Jacobi preconditioner)   and
#            JAC (solver::Ir + local Jacobi, relaxation 0.9 -- damped Jacobi sweeps)
#   iters :  {1, 5, 10, 15, 20, 25, 50}   (fixed Iteration count, no residual tolerance)
#
# i.e. 2 x 7 = 14 runs, each a 30-step march from t0 with solver caching ON (cacheSolver=true,
# preconditionerRebuildInterval=REBUILD, default 100) -- matching the precfloat best-practice run,
# so the only variable across the study is the coarse-grid solve. The coarse solver runs on each
# rank's LOCAL coarse matrix (the MG already lives inside the outer localized Schwarz), so it is a
# plain local solver -- NOT Schwarz-wrapped.
#
# Run names (under paramStudyResults/coarse-solve/):
#   pMG-precfloat-coarse-<cg|jacobi><iters>-cache-rebuild<REBUILD>
# The "-cache-rebuild<N>" infix keeps each run in report_cache_reuse's reuse/rebuild tally.
#
# The 10 coarse-solver configs are generated here (from the precfloat base) into system/gko/ as
#   p-multigrid-localized-precfloat-coarse-<solver><iters>.json
# so the study is self-contained; they are overwritten on each run.
#
# Tunables (env overrides):
#   BASE_CFG=p-multigrid-localized-precfloat.json   the float-MG base whose coarsest_solver is swept
#   COARSE_SOLVERS="cg jacobi"                       which coarse solvers to sweep
#   COARSE_ITERS="1 5 10 15 20 25 50"                coarse-solver iteration counts
#   REBUILD=100                                      preconditionerRebuildInterval (solver cache)
#   KOKKOS_TOOL=<name>                               load a kokkos-tools profiler for each run
#
# Usage:   ./param-study-mg-coarse-solve.sh                 # full 2x7 sweep
#          ./param-study-mg-coarse-solve.sh cg              # only the CG coarse-solver sweep
#          ./param-study-mg-coarse-solve.sh jacobi          # only the Jacobi coarse-solver sweep
#          COARSE_ITERS="10 50" ./param-study-mg-coarse-solve.sh
#          KOKKOS_TOOL=space-time-stack ./param-study-mg-coarse-solve.sh   # profiled

STUDY_TYPE="${STUDY_TYPE:-coarse-solve}"
source "$(dirname "$0")/param-study-mg-common.sh"

ensure_mg_variants   # ensures the precfloat base config exists

BASE_CFG="${BASE_CFG:-p-multigrid-localized-precfloat.json}"
REBUILD="${REBUILD:-100}"
COARSE_ITERS=(${COARSE_ITERS:-1 5 10 15 20 25 50})
# Positional args (if any) select the coarse solvers; otherwise sweep both.
if [ "$#" -gt 0 ]; then
    COARSE_SOLVERS=("$@")
else
    COARSE_SOLVERS=(${COARSE_SOLVERS:-cg jacobi})
fi

if [ ! -f "system/gko/$BASE_CFG" ]; then
    echo "!! base config 'system/gko/$BASE_CFG' not found -- cannot run coarse-solve study"; exit 1
fi

# Write a coarse-solver variant of BASE_CFG: deep-copy the base and replace the multigrid
# coarsest_solver (preconditioner.local_solver.coarsest_solver) with a fixed-iteration CG or
# damped-Jacobi LOCAL solver. $1=solver(cg|jacobi) $2=iters $3=output basename in system/gko/.
gen_coarse_cfg() {
    local solver="$1" n="$2" out="$3"
    python3 - "system/gko/$BASE_CFG" "$solver" "$n" "system/gko/$out" <<'PY'
import json, sys
base, solver, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = json.load(open(base))
if solver == "cg":
    # solver::Cg with a LOCAL Jacobi preconditioner (no Schwarz -- the coarse matrix is local),
    # run for exactly n iterations.
    coarse = {
        "type": "solver::Cg",
        "preconditioner": {"type": "preconditioner::Jacobi", "max_block_size": 1},
        "criteria": [{"type": "Iteration", "max_iters": n}],
    }
elif solver == "jacobi":
    # damped-Jacobi sweeps via solver::Ir (relaxation 0.9), n iterations.
    coarse = {
        "type": "solver::Ir",
        "relaxation_factor": 0.9,
        "solver": {"type": "preconditioner::Jacobi", "max_block_size": 1},
        "criteria": [{"type": "Iteration", "max_iters": n}],
    }
else:
    sys.exit("!! unknown coarse solver '%s' (use cg or jacobi)" % solver)
try:
    cfg["preconditioner"]["local_solver"]["coarsest_solver"] = coarse
except (KeyError, TypeError):
    sys.exit("!! base config has no preconditioner.local_solver.coarsest_solver to replace")
with open(out, "w") as f:
    json.dump(cfg, f, indent=4)
    f.write("\n")
PY
}

echo "################################################################"
echo " coarse-grid-solver sweep on $BASE_CFG (float MG preconditioner)"
echo "   solvers: ${COARSE_SOLVERS[*]}   iters: ${COARSE_ITERS[*]}   rebuild: $REBUILD"
echo "################################################################"

for solver in "${COARSE_SOLVERS[@]}"; do
    case "$solver" in
        cg|jacobi) ;;
        *) echo "!! skipping unknown coarse solver '$solver' (use cg or jacobi)"; continue ;;
    esac
    for n in "${COARSE_ITERS[@]}"; do
        out="p-multigrid-localized-precfloat-coarse-${solver}${n}.json"
        if ! gen_coarse_cfg "$solver" "$n" "$out"; then
            echo "!! failed to generate $out -- skipping"; continue
        fi
        # run_cache_interval injects cacheSolver=true + preconditionerRebuildInterval=$REBUILD and
        # names the run pMG-<label>-cache-rebuild<REBUILD>.
        run_cache_interval "precfloat-coarse-${solver}${n}" "$out" "$REBUILD"
    done
done

print_summary
report_cache_reuse
