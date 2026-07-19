#!/bin/bash
#SBATCH --job-name=phase3g-paper-mgsc-coarse-iters
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase3g-paper-study-mgsc-coarse-iters.sh
# Or run live: ./phase3g-paper-study-mgsc-coarse-iters.sh   (subset: ./...sh c2 c4)
# Override:    STEPS=50   MAXLEVELS=5
#
# PHASE 3g: coarsest-solver max_iters sweep for the GLOBAL scale-corrected MG.
# Base = p-multigrid-mgsc-merge2-lcg.json (the §4.11j best building blocks):
#   solver::Cg + Multigrid(scale_correction=true) + pgmMerge2 + localized-CG coarse
#   (coarsest_solver = Schwarz{ Cg + Jacobi }), max_levels pinned at the §4.11j knee (L5).
#
# The coarsest Cg runs a FIXED max_iters with NO residual criterion -> it always burns the full count
# (no early exit). Base config uses 10; §4.11e/i suggested lean coarse counts barely move the needle
# once the coarse solve is LOCALIZED (tiny coarse block, min_coarse_rows=64). This sweep quantifies it:
# does dropping the coarse Cg to 1/2/4/6 iters cost outer CG iterations, or is it free wall time?
#   c10 (base) reference: §4.11j L5 = 7.5 outer iters, p_ms 1117, 3.660 s/step.
#
# Sweeps ONLY coarsest_solver.local_solver.criteria[0].max_iters via jq (targeted -- there are several
# "max_iters" in the file: pre/post-smoother=1, multigrid=1, outer=150; jq path avoids clobbering them).
# max_levels fixed (MAXLEVELS, default 5); pre/post smoother stay 1 sweep each.
#
# Read (SUMMARY): p_iters (does the outer CG need more V-cycles as the coarse solve weakens?),
# p_ms/solve, s/step (best cell), cont (~3e-6). All CACHED (cacheSolver=true, interval=0).
#
# Requires the Phase-0 restart (processor*/$RESTART/); aborts loudly if missing.

STUDY_TYPE=mgsc-coarse-iters
STEPS="${STEPS:-50}"
MAXLEVELS="${MAXLEVELS:-5}"     # pin at the §4.11j knee
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

BASE_CFG="system/gko/p-multigrid-mgsc-merge2-lcg.json"   # global mgsc + pgmMerge2 + localized-CG coarse
SWEEP_CFG="system/gko/mgsccoarse-papertmp.json"
cleanup_sweep_cfg() { rm -f "$SWEEP_CFG"; }
trap cleanup_sweep_cfg EXIT   # chains with common.sh's restore() trap

build_variant_config() {
    # $1 = coarse max_iters. Clone base, pin max_levels, set ONLY the coarsest Cg's max_iters (jq path).
    local citers="$1"
    if [ ! -f "$BASE_CFG" ]; then echo "!! missing base config $BASE_CFG" >&2; return 1; fi
    jq --argjson ml "$MAXLEVELS" --argjson ci "$citers" \
       '.preconditioner.max_levels = $ml
        | .preconditioner.coarsest_solver.local_solver.criteria[0].max_iters = $ci' \
       "$BASE_CFG" > "$SWEEP_CFG" || return 1
    local got
    got=$(jq '.preconditioner.coarsest_solver.local_solver.criteria[0].max_iters' "$SWEEP_CFG")
    [ "$got" = "$citers" ] || { echo "!! failed to set coarse max_iters=$citers (got $got)" >&2; return 1; }
}

build_variant_fvsolution() {
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${SWEEP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

run_variant() {
    local name="$1" citers="$2" desc="$3"
    build_variant_config "$citers" || { echo "   skip $name"; return; }
    build_variant_fvsolution       || { echo "   skip $name"; return; }
    run_one "$name" "$TMP_FVSOL" "$desc"
}

declare -A VARIANT
ORDER=()
for C in 1 2 4 6 8; do
    name="c${C}"
    VARIANT[$name]="$C | GLOBAL Cg+MG mgsc, CACHED, pgmMerge2, localized-CG coarse (max_iters=$C), max_levels=$MAXLEVELS"
    ORDER+=("$name")
done

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "Phase 3g (global mgsc + merge2 + localized-CG coarse; max_levels=$MAXLEVELS): $STEPS iters/variant from restart $RESTART; coarse max_iters: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    citers="${spec%% |*}"; desc="${spec#*| }"
    run_variant "$name" "$citers" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   find the coarse max_iters knee (localized coarse Cg, no early exit)"
echo "  reference: §4.11j L5 (coarse max_iters=10) = 7.5 outer iters, 3.660 s/step"
echo "phase3g done"
