#!/bin/bash
# Does scale correction port to the LOCALIZED MG if we drop the useless pre pass?
#
# The prior verdict -- "sc does not port to localized, 9x slower" (the localized-scalecorr dead-end) --
# was measured with BOTH passes. Two things changed since:
#   1. The pre pass is now known to contribute ZERO convergence (14.0 iters vs 13.8 for no sc at all),
#      so half of what was tested was pure overhead.
#   2. Inside a localized MG the whole hierarchy is rank-local, so sc's A*delta SpMV AND its Rayleigh
#      dots are LOCAL -> sc would cost ZERO communication here. On the global branch sc lost precisely
#      because it was 55% of per-V-cycle comm; that cost does not exist in this architecture.
#
# RISK (why it may still fail): sf makes the preconditioner NONLINEAR (it depends on the vector being
# preconditioned), and per-rank sf makes it spatially DISCONTINUOUS across subdomain boundaries. Cg
# assumes a FIXED LINEAR SPD preconditioner. That is the likely cause of the original 9x, and post-only
# halves the offending operations without removing the discontinuity. Hence a test, not a prediction.
#
# At the localized champion cell: L8, coarse rel-tol 0.25 (= 2.236 s/step, 26.7 iters).
STUDY_TYPE=localized-sc
STEPS="${STEPS:-50}"
export MPIRUN_FORWARD_ENV="NEON_MGSC_MODE"
source ./paper-study-common.sh
require_restart

BASE=system/gko/p-multigrid-localized-solver-merge2.json
CFG=system/gko/locsc-papertmp.json
trap 'rm -f "$CFG"' EXIT

build() {   # $1 = scale_correction bool
    jq --argjson sc "$1" '
        .preconditioner.local_solver.max_levels = 8
        | .preconditioner.local_solver.scale_correction = $sc
        | .preconditioner.local_solver.coarsest_solver.criteria = [
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":0.25},
            {"type":"Iteration","max_iters":50}
          ]' "$BASE" > "$CFG" || return 1
    jq -e ".preconditioner.local_solver.scale_correction == $1" "$CFG" >/dev/null || return 1
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

# baseline: reproduces the localized champion (2.236 / 26.7) -- also proves the sc path is wired
build false; unset NEON_MGSC_MODE
run_one "loc-sc-none" "$TMP_FVSOL" "localized MG L8/tol0.25, scale_correction OFF (champion baseline)"

# the question: post pass only -- sc's mechanism at ZERO communication cost
build true; export NEON_MGSC_MODE=post
run_one "loc-sc-post" "$TMP_FVSOL" "localized MG L8/tol0.25, sc POST PASS ONLY (local dots, zero comm)"

# control: does the prior 9x dead-end reproduce with both passes?
build true; export NEON_MGSC_MODE=both
run_one "loc-sc-both" "$TMP_FVSOL" "localized MG L8/tol0.25, sc BOTH passes (prior verdict: 9x slower)"

unset NEON_MGSC_MODE
print_summary
echo "localized-sc done"
