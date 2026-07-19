#!/bin/bash
#SBATCH --job-name=phase3e-paper-mergelevels-maxlevels-scfwd
#SBATCH --nodelist=gpu-nvidia-h200-2
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase3e-paper-study-mergelevels-maxlevels-scfwd.sh
# Or run live: salloc ... then ./phase3e-paper-study-mergelevels-maxlevels-scfwd.sh
# Override:    STEPS=50 ; run a subset by name, e.g. ./phase3e-...sh ml2_L6 ml4_L10
#
# Optimization-paper study -- PHASE 3e: mergeLevels x max_levels depth sweep, WITH FORWARD SMOOTHER
# SCALE CORRECTION. Identical grid to Phase 3d (phase3d-paper-study-mergelevels-maxlevels.sh) but the
# PRE-smoother's Ir scale_correction is flipped "none" -> "forward" (down-sweep Rayleigh correction);
# post-smoother stays "none" and the coarsest solver stays "forward" (already so in the base configs).
# This is the phase3b "fwd" recipe (see phase3b-paper-study-scalecorr.sh) grafted onto the MergedPgm
# depth sweep -- the per-smoother scale_correction MODE (gko::solver::scale_correction_mode), NOT the
# MG-level `scale_correction: true` boolean and NOT mixed precision.
#
#   scale_correction layout per generated config:
#       pre_smoother    : forward   (<- flipped from the Phase-3d "none")
#       post_smoother   : none
#       coarsest_solver : forward   (unchanged base value)
#
# !! CAVEAT -- KNOWN DEAD-END COMBINATION. memory occdrivaer-scalecorr-localized-deadend records that
#    per-smoother scale correction + the LOCALIZED Schwarz{Multigrid} path (which every config here
#    uses) was previously 9x slower (1709 s, 446 iters, 94% in pEqn): the per-subdomain Rayleigh
#    scaling is inconsistent across the decomposition and convergence collapses. This sweep RE-ENTERS
#    that combination on purpose, to test it against the merge/depth grid -- expect some or all
#    variants to converge poorly or blow up. Read the SUMMARY p_iters/cont before trusting any s/step.
#
# The grid (same as 3d): mergeLevels in {2,4} x max_levels in {2,4,6,8,10}; for MergedPgm max_levels
# counts MERGED levels so effective Pgm depth ~= mergeLevels * max_levels. All CACHED
# (cacheSolver=true, preconditionerRebuildInterval=0). Results go to a DEDICATED dir so they never mix
# with the Phase-3d (scale-correction-off) baseline:
#       paperParamStudyResults/mergelevels-maxlevels-scfwd/
#
# Compare EACH variant here against its Phase-3d twin (same name, mergelevels-maxlevels/ dir): does
# forward correction cut p_iters enough to beat the no-correction depth sweep, or does the localized
# decomposition poison it as the dead-end predicts?
#
# Requires the Phase-0 restart (processor*/$RESTART/); aborts loudly if missing.

STUDY_TYPE=mergelevels-maxlevels-scfwd
STEPS="${STEPS:-50}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"

require_restart

# Distinct temp-config basename so a concurrent Phase-3d run (which uses mergesweep-papertmp.json)
# is never clobbered. Plain basename -- a leading-dot component breaks OpenFOAM's dict parser.
SWEEP_CFG="system/gko/mergesweep-scfwd-papertmp.json"
cleanup_sweep_cfg() { rm -f "$SWEEP_CFG"; }
trap cleanup_sweep_cfg EXIT   # chains with common.sh's restore() trap.

build_variant_config() {
    # $1 = mergeLevels (2|4), $2 = max_levels. Clone the committed merge<N> base config, rewriting the
    # "max_levels" entry AND flipping the FIRST scale_correction "none" (the pre-smoother) to "forward".
    # post-smoother "none" and coarsest "forward" are left as-is. awk edits both in one pass; the `sc`
    # flag guarantees ONLY the first (pre) none is touched.
    local merge="$1" maxlevels="$2"
    local base="system/gko/p-multigrid-localized-solver-merge${merge}.json"
    if [ ! -f "$base" ]; then echo "!! missing base config $base" >&2; return 1; fi
    awk -v MAXL="$maxlevels" '
        /"max_levels"[[:space:]]*:/ { sub(/[0-9]+/, MAXL) }
        !sc && /"scale_correction"[[:space:]]*:[[:space:]]*"none"/ { sub(/"none"/, "\"forward\""); sc=1 }
        { print }
    ' "$base" > "$SWEEP_CFG" || return 1
    # Guards: max_levels took, and the layout is exactly pre=forward/post=none/coarsest=forward
    # (=> 2 "forward", 1 "none"). Fail loudly if the base schema drifted.
    grep -qE "\"max_levels\"[[:space:]]*:[[:space:]]*${maxlevels}\b" "$SWEEP_CFG" \
        || { echo "!! failed to set max_levels=$maxlevels in $base" >&2; return 1; }
    local nfwd nnone
    nfwd=$(grep -cE "\"scale_correction\"[[:space:]]*:[[:space:]]*\"forward\"" "$SWEEP_CFG")
    nnone=$(grep -cE "\"scale_correction\"[[:space:]]*:[[:space:]]*\"none\"" "$SWEEP_CFG")
    if [ "$nfwd" -ne 2 ] || [ "$nnone" -ne 1 ]; then
        echo "!! unexpected scale_correction layout (forward=$nfwd none=$nnone; expected 2/1)" >&2
        return 1
    fi
}

build_variant_fvsolution() {
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${SWEEP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

run_variant() {
    local name="$1" merge="$2" maxlevels="$3" desc="$4"
    build_variant_config "$merge" "$maxlevels"   || { echo "   skip $name"; return; }
    build_variant_fvsolution                     || { echo "   skip $name"; return; }
    run_one "$name" "$TMP_FVSOL" "$desc"
}

declare -A VARIANT
ORDER=()
for merge in 2 4; do
    for L in 2 4 6 8 10; do
        name="ml${merge}_L${L}"
        VARIANT[$name]="$merge $L | localized MG, CACHED, MergedPgm mergeLevels=$merge, max_levels=$L, scale_correction pre=forward (eff ~$((merge*L)) Pgm steps)"
        ORDER+=("$name")
    done
done

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "Phase 3e: $STEPS iterations/variant from restart $RESTART; FORWARD pre-smoother scale correction; variants: ${SEL[*]}"
echo "   (see occdrivaer-scalecorr-localized-deadend: scalecorr + localized Schwarz was previously 9x slower)"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    params="${spec%% |*}"; desc="${spec#*| }"
    read -r merge maxlevels _ <<< "$params"
    run_variant "$name" "$merge" "$maxlevels" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   compare EACH variant to its Phase-3d twin (mergelevels-maxlevels/);"
echo "watch p_iters/cont -- forward scale correction on the localized path may collapse convergence."
