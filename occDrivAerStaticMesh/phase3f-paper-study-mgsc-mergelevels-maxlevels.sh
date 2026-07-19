#!/bin/bash
#SBATCH --job-name=phase3f-paper-mgsc-mergelevels-maxlevels
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase3f-paper-study-mgsc-mergelevels-maxlevels.sh
# Or run live: ./phase3f-paper-study-mgsc-mergelevels-maxlevels.sh   (subset: ./...sh L4 L6)
# Override:    STEPS=50
#
# PHASE 3f: max_levels sweep for the GLOBAL (non-localized) scale-corrected MG.
# This is the NON-LOCALIZED analog of phase3d (which was the localized Schwarz{MG} path). Config:
#   base = p-multigrid-mgsc-merge2-lcg.json =
#     solver::Cg  (global, over the whole distributed matrix)
#     + solver::Multigrid, scale_correction=true (MG-level Rayleigh scaling, §4.11b, the win)
#     + mg_level neon::pgmMerge2 (mergeLevels=2)
#     + coarsest_solver = preconditioner::Schwarz{ solver::Cg(10) + Jacobi }  (LOCALIZED coarse solve,
#       §4.11i -- kills the coarse-grid halo exchanges: each rank solves its local coarse block, no
#       distributed SpMV per coarse iteration; confirmed 821->5 sub-1KB Alltoallv, ~3.6% faster at L5).
#
# Sweeps only max_levels; mergeLevels is fixed at 2 and the localized-CG coarse solver is fixed. The
# question: for the global mgsc+merge2 config with the coarse halos removed, where is the max_levels
# knee? (§4.11e found L4-L5 optimal for merge3 with the OLD global coarse solver; merge2 coarsens less
# aggressively per level and the coarse comm is now gone, so the knee may move -- this re-finds it.)
#
# All CACHED (cacheSolver=true, interval=0). max_levels counts MERGED levels (each pgmMerge2 jumps 2 Pgm
# steps); the hierarchy still bottoms out at min_coarse_rows=64, so large max_levels may build fewer.
#
# Read (SUMMARY): s/step (best cell), p_iters (does convergence hold as depth drops?), cont (~3e-6).
# Reference points: §4.11e best L5c4 (merge3, OLD coarse) 3.84; §4.11i L5c4-lcg (merge3, localized coarse) 3.74.
#
# Requires the Phase-0 restart (processor*/$RESTART/); aborts loudly if missing.

STUDY_TYPE=mgsc-mergelevels-maxlevels
STEPS="${STEPS:-50}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

BASE_CFG="system/gko/p-multigrid-mgsc-merge2-lcg.json"  # global mgsc + pgmMerge2 + localized-CG coarse
MERGE=2                                                  # mergeLevels fixed at 2
SWEEP_CFG="system/gko/mgscsweep-papertmp.json"
cleanup_sweep_cfg() { rm -f "$SWEEP_CFG"; }
trap cleanup_sweep_cfg EXIT   # chains with common.sh's restore() trap

build_variant_config() {
    # $1 = max_levels. Clone the base config rewriting only its "max_levels": N entry.
    local maxlevels="$1"
    if [ ! -f "$BASE_CFG" ]; then echo "!! missing base config $BASE_CFG" >&2; return 1; fi
    sed -E "s#(\"max_levels\"[[:space:]]*:[[:space:]]*)[0-9]+#\1${maxlevels}#" "$BASE_CFG" > "$SWEEP_CFG" \
        || return 1
    grep -qE "\"max_levels\"[[:space:]]*:[[:space:]]*${maxlevels}\b" "$SWEEP_CFG" \
        || { echo "!! failed to set max_levels=$maxlevels in $BASE_CFG" >&2; return 1; }
}

build_variant_fvsolution() {
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${SWEEP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

run_variant() {
    local name="$1" maxlevels="$2" desc="$3"
    build_variant_config "$maxlevels" || { echo "   skip $name"; return; }
    build_variant_fvsolution          || { echo "   skip $name"; return; }
    run_one "$name" "$TMP_FVSOL" "$desc"
}

declare -A VARIANT
ORDER=()
for L in 2 3 4 5 6 8 10; do
    name="L${L}"
    VARIANT[$name]="$L | GLOBAL Cg+MG mgsc, CACHED, pgmMerge2, localized-CG coarse, max_levels=$L (eff ~$((MERGE*L)) Pgm steps)"
    ORDER+=("$name")
done

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "Phase 3f (global mgsc + merge2 + localized-CG coarse): $STEPS iters/variant from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    maxlevels="${spec%% |*}"; desc="${spec#*| }"
    run_variant "$name" "$maxlevels" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   find the max_levels knee for global mgsc+merge2 with the coarse halos removed"
echo "  reference: §4.11e L5(merge3,old coarse) 3.84 ; §4.11i L5-lcg(merge3,localized coarse) 3.74"
