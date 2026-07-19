#!/bin/bash
#SBATCH --job-name=investigate-mgsc-levels-coarse
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# 2-D sweep: max_levels x coarsest-solver iterations, on the GLOBAL MG WITH MG-level scale correction
# AND the optimal mergeLevel (pgmMerge3). This gap was never covered: every prior level/coarse sweep
# was either scale-correction OFF (p-multigrid.L*.sc0, mergelevels) or localized+precfloat (PMIS,
# precfloat coarse-cg grid) -- none on the global MG-level-sc regime.
#
# Motivation: MG-level sc collapsed the outer pressure iteration count 16.5 -> ~6-9 (§4.11b/d). At so
# few V-cycles per solve the cost balance shifts -- the deep-level dispatch tail (§4.7) and the coarse
# solve are each a LARGER fraction of a solve than at 16 iters -- so the max_levels / coarse-iter
# optimum found under no-sc need not hold here. Base config = p-multigrid-mgsc-merge3.json (Cg + global
# MG + scale_correction=true + neon::pgmMerge3, the §4.11d best cell, 4.24 s/step).
#
#   grid (12 cells):  max_levels in {4,6,8,10}  x  coarsest_solver max_iters in {4,8,16}
#   variant name  Lc<max_levels>i<coarse_iters>   config p-multigrid-mgsc-m3-L<ml>-c<ci>.json
#
# NB max_levels counts MERGED levels (each pgmMerge3 level jumps 3 Pgm steps); the hierarchy still
# bottoms out at min_coarse_rows=64, so large L may build fewer levels than requested -- the setup log
# reports the actual count. All CACHED (cacheSolver=true, interval=0), 50 steps from the iter-1000
# restart, same session (single-run variance ~20%; trust p_iters + monotone trends over small s/step).
#
# Read: p_iters (does the ~6-9 convergence survive as levels drop / coarse weakens?), p_ms/solve,
#       s/step (best cell), cont (~3e-6, watch for divergence at shallow L + weak coarse).
# Reference: the current production point is L10/c8 == mgsc-m3 (§4.11d, ~4.24 s/step).

STUDY_TYPE=mgsc-levels-coarse
STEPS="${STEPS:-50}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"

require_restart

build_variant_fvsolution() {
    local cfg="$1" cache="true" interval="0"
    if [ ! -f "system/gko/$cfg" ]; then echo "!! missing system/gko/$cfg" >&2; return 1; fi
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$cfg;\n        cacheSolver      $cache;\n        preconditionerRebuildInterval $interval;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}
run_variant() { build_variant_fvsolution "$2" || { echo "   skip $1"; return; }; run_one "$1" "$TMP_FVSOL" "$3"; }

# Build the 12-cell grid programmatically (name -> config + desc).
declare -A VARIANT=()
ORDER=()
for L in 2 3 4 5 6 8 10; do
    for C in 4 6 8 10 16; do
        name="L${L}c${C}"
        VARIANT[$name]="p-multigrid-mgsc-m3-L${L}-c${C}.json | mgsc+pgmMerge3, CACHED, max_levels=${L}, coarse_iters=${C}"
        ORDER+=("$name")
    done
done

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "mgsc levels x coarse: $STEPS iterations/variant from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    cfg="${spec%% *}"; desc="${spec#*| }"
    run_variant "$name" "$cfg" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   2-D: rows=max_levels{4,6,8,10}, cols=coarse_iters{4,8,16}; ref=L10c8 (mgsc-m3, 4.24)"
