#!/bin/bash
#
# LEVEL x COARSE-CG-ITERS study for occDrivAerStaticMesh, on the new best-practice precfloat setup
# (FLOAT MG preconditioner).
#
# Base config: system/gko/p-multigrid-localized-precfloat.json -- the localized
# Schwarz{Multigrid(local)} preconditioner with "value_type":"float32" (the validated float MG
# preconditioner; outer solver::Cg and the l1ScaledResidual stop stay fp64). Everything is held
# fixed EXCEPT two knobs, swept as a 2D grid:
#
#   max_levels : {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}   the inner Multigrid's max_levels
#   coarse cg  : {1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20} coarsest-solver CG iteration count
#
# i.e. 10 x 11 = 110 runs, each a 30-step march from t0 with solver caching ON (cacheSolver=true,
# preconditionerRebuildInterval=REBUILD, default 100) -- matching the precfloat best-practice run,
# so the only variables across the study are the V-cycle depth and the coarse-grid solve. The
# coarsest solver is a fixed-iteration solver::Cg + LOCAL Jacobi preconditioner (no Schwarz -- the
# coarse matrix is already local, inside the outer localized Schwarz).
#
# Run names (under paramStudyResults/level-sweep/):
#   pMG-precfloat-L<lev>-coarsecg<iters>-cache-rebuild<REBUILD>
# The "-cache-rebuild<N>" infix keeps each run in report_cache_reuse's reuse/rebuild tally.
#
# The 110 config variants are generated here (from the precfloat base) into system/gko/ as
#   p-multigrid-localized-precfloat-L<lev>-coarsecg<iters>.json
# so the study is self-contained; they are overwritten on each run.
#
# Tunables (env overrides):
#   BASE_CFG=p-multigrid-localized-precfloat.json   the float-MG base whose max_levels/coarse are swept
#   LEVELS="2 4 6 8 10 12 14 16 18 20"               inner Multigrid max_levels values
#   COARSE_ITERS="1 2 4 6 8 10 12 14 16 18 20"        coarsest-solver CG iteration counts
#   REBUILD=100                                      preconditionerRebuildInterval (solver cache)
#   KOKKOS_TOOL=<name>                               load a kokkos-tools profiler for each run
#
# Usage:   ./param-study-mg-level-sweep.sh                       # full 10x11 sweep
#          LEVELS="6 10" ./param-study-mg-level-sweep.sh         # only two level values
#          COARSE_ITERS="1 8" ./param-study-mg-level-sweep.sh    # only two coarse-iter values
#          KOKKOS_TOOL=space-time-stack ./param-study-mg-level-sweep.sh   # profiled

STUDY_TYPE="${STUDY_TYPE:-level-sweep}"
source "$(dirname "$0")/param-study-mg-common.sh"

ensure_mg_variants   # ensures the precfloat base config exists

BASE_CFG="${BASE_CFG:-p-multigrid-localized-precfloat.json}"
REBUILD="${REBUILD:-100}"
LEVELS=(${LEVELS:-2 4 6 8 10 12 14 16 18 20})
COARSE_ITERS=(${COARSE_ITERS:-1 2 4 6 8 10 12 14 16 18 20})

if [ ! -f "system/gko/$BASE_CFG" ]; then
    echo "!! base config 'system/gko/$BASE_CFG' not found -- cannot run level-sweep study"; exit 1
fi

# Write a level/coarse-iters variant of BASE_CFG: deep-copy the base, set the inner Multigrid's
# max_levels, and replace the multigrid coarsest_solver (preconditioner.local_solver.coarsest_solver)
# with a fixed-iteration solver::Cg + LOCAL Jacobi coarse solver.
# $1=max_levels $2=coarse cg iters $3=output basename in system/gko/.
gen_level_cfg() {
    local lev="$1" n="$2" out="$3"
    python3 - "system/gko/$BASE_CFG" "$lev" "$n" "system/gko/$out" <<'PY'
import json, sys
base, lev, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = json.load(open(base))
try:
    local = cfg["preconditioner"]["local_solver"]
except (KeyError, TypeError):
    sys.exit("!! base config has no preconditioner.local_solver to configure")
local["max_levels"] = lev
# solver::Cg with a LOCAL Jacobi preconditioner (no Schwarz -- the coarse matrix is local),
# run for exactly n iterations.
local["coarsest_solver"] = {
    "type": "solver::Cg",
    "preconditioner": {"type": "preconditioner::Jacobi", "max_block_size": 1},
    "criteria": [{"type": "Iteration", "max_iters": n}],
}
with open(out, "w") as f:
    json.dump(cfg, f, indent=4)
    f.write("\n")
PY
}

echo "################################################################"
echo " max_levels x coarse-CG-iters sweep on $BASE_CFG (float MG preconditioner)"
echo "   levels: ${LEVELS[*]}"
echo "   coarse cg iters: ${COARSE_ITERS[*]}   rebuild: $REBUILD"
echo "   -> $(( ${#LEVELS[@]} * ${#COARSE_ITERS[@]} )) runs"
echo "################################################################"

for lev in "${LEVELS[@]}"; do
    for n in "${COARSE_ITERS[@]}"; do
        out="p-multigrid-localized-precfloat-L${lev}-coarsecg${n}.json"
        if ! gen_level_cfg "$lev" "$n" "$out"; then
            echo "!! failed to generate $out -- skipping"; continue
        fi
        # run_cache_interval injects cacheSolver=true + preconditionerRebuildInterval=$REBUILD and
        # names the run pMG-<label>-cache-rebuild<REBUILD>.
        run_cache_interval "precfloat-L${lev}-coarsecg${n}" "$out" "$REBUILD"
    done
done

print_summary
report_cache_reuse
