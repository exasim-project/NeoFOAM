#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
while ! grep -q "localized-smoother all done" "$OUT/smoother.output" 2>/dev/null; do sleep 30; done
echo "=== smoother done; localized MG + scale correction test ==="

STUDY_TYPE=localized-sc
STEPS=50
export MPIRUN_FORWARD_ENV="NEON_MGSC_MODE"
source ./paper-study-common.sh
require_restart

BASE=system/gko/p-multigrid-localized-solver-merge2.json
CFG=system/gko/locsc-papertmp.json
trap 'rm -f "$CFG"' EXIT

# At the localized optimum: L8, coarse rel-tol 0.25 (§5.1). scale_correction lives on the LOCAL
# multigrid (inside Schwarz) -> its A.delta SpMV and Rayleigh dots are all RANK-LOCAL, so unlike the
# global branch (where sc = 55% of per-V-cycle comm) it costs ZERO communication here -- pure
# arithmetic, on a GPU that is ~80% idle.
# RISK: sf makes the preconditioner NONLINEAR (it depends on the vector being preconditioned), and
# localized it is spatially DISCONTINUOUS -- each rank computes its own sf, so the correction jumps at
# subdomain boundaries. Cg assumes a fixed linear SPD preconditioner. The prior verdict (both passes:
# 9x slower, the localized-scalecorr dead-end) is almost certainly this. Post-only halves the offending
# operations but does not remove the discontinuity -- hence a test, not a prediction.
build() {   # $1 = scale_correction bool
    jq --argjson sc "$1" '
        .preconditioner.local_solver.max_levels = 8
        | .preconditioner.local_solver.scale_correction = $sc
        | .preconditioner.local_solver.coarsest_solver.criteria = [
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":0.25},
            {"type":"Iteration","max_iters":50}
          ]' "$BASE" > "$CFG"
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL"
}

# baseline (sc off) -- reproduces the §5.1 localized champion, 2.236 s/step / 26.7 iters
build false; unset NEON_MGSC_MODE
run_one "loc-sc-none" "$TMP_FVSOL" "localized MG L8/tol0.25, scale_correction OFF (champion baseline)"

# post-only -- the question
build true; export NEON_MGSC_MODE=post
run_one "loc-sc-post" "$TMP_FVSOL" "localized MG L8/tol0.25, scale correction POST PASS ONLY (local dots, zero comm)"

# both -- control: does the prior 9x dead-end reproduce?
build true; export NEON_MGSC_MODE=both
run_one "loc-sc-both" "$TMP_FVSOL" "localized MG L8/tol0.25, scale correction BOTH passes (prior: 9x slower)"

unset NEON_MGSC_MODE
print_summary
echo "localized-sc done"
