#!/bin/bash
# Does the FAITHFUL scale-correction port flip sc from a loss to a win?
#
# The port now matches OpenFOAM's GAMGSolver::scale on the expensive axis: the inner operator is
# D^-1 (Ir's inner Schwarz{Jacobi}) applied directly, instead of a full `pre_smoother->apply()` that
# pays an EXTRA residual SpMV -- a distributed halo per correction point per level per V-cycle that
# OpenFOAM never spends. SpMV halos were 77% of sc's measured communication (+82 of +107 ms/V-cycle).
#
# Reference numbers with the UNFAITHFUL port, same cell (L4/merge2/localized-coarse/tol0.1, sc off
# baseline = 3.360 naive):
#     sc=both 3.640 (7.7 iters) | sc=post 3.480 (9.3) | sc=none 3.380 (13.8)
# sc lost by 7.7%. If removing one of its two SpMVs per pass is worth more than that, sc flips.
#
# Convergence MUST be ~unchanged (the inner op is still D^-1, modulo Ir's dropped 0.9 relaxation --
# OpenFOAM's scale() applies no relaxation, so dropping it is the faithful choice). If iters move a
# lot, the port changed the math, not just the cost.
STUDY_TYPE=faithful-sc
STEPS="${STEPS:-50}"
export MPIRUN_FORWARD_ENV="NEON_MGSC_MODE"
source ./paper-study-common.sh
require_restart
BASE=system/gko/p-multigrid-mgsc-merge2-lcg.json
CFG=system/gko/faithsc-papertmp.json
trap 'rm -f "$CFG"' EXIT
build() {
    jq --argjson sc "$1" '
        .preconditioner.max_levels = 4
        | .preconditioner.scale_correction = $sc
        | .preconditioner.coarsest_solver.local_solver.criteria = [
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":0.1},
            {"type":"Iteration","max_iters":50}]' "$BASE" > "$CFG"
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL"
}
build false; unset NEON_MGSC_MODE
run_one "faith-none" "$TMP_FVSOL" "L4/tol0.1 sc OFF (baseline; unaffected by the port)"
build true; export NEON_MGSC_MODE=post
run_one "faith-post" "$TMP_FVSOL" "L4/tol0.1 sc POST only, FAITHFUL port (was 3.480)"
build true; export NEON_MGSC_MODE=both
run_one "faith-both" "$TMP_FVSOL" "L4/tol0.1 sc BOTH, FAITHFUL port (was 3.640)"
unset NEON_MGSC_MODE
print_summary
echo "faithful-sc done"
