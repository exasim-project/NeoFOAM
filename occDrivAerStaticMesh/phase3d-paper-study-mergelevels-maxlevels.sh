#!/bin/bash
#SBATCH --job-name=phase3d-paper-mergelevels-maxlevels
#SBATCH --nodelist=gpu-nvidia-h200-2
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase3d-paper-study-mergelevels-maxlevels.sh
# Or run live: salloc ... then ./phase3d-paper-study-mergelevels-maxlevels.sh
# Override:    STEPS=50 ; run a subset by name, e.g. ./phase3d-...sh ml2_L6 ml4_L10
#
# Optimization-paper study -- PHASE 3d: mergeLevels x max_levels interaction (MergedPgm depth sweep).
#
# Phase 3c (phase3c-paper-study-mergelevels.sh) showed that at the DEFAULT max_levels=10, merging
# Pgm steps buys nothing: the per-iteration dispatch savings from a shallower V-cycle are almost
# exactly cancelled by the convergence penalty of coarser transfer (avg p-iters climbed 23.5 -> 35.5
# for mergeLevels 1 -> 4; wall time flat ~155 s, net slower at ml4). See memory mergelevels-no-speedup.
#
# That leaves one lever unexplored: max_levels was pinned at 10 for every merge factor, so the actual
# hierarchy DEPTH varied wildly. For MergedPgm, max_levels counts MERGED levels, and each merged level
# jumps `mergeLevels` Pgm steps -- so effective Pgm depth ~= mergeLevels * max_levels:
#
#              max_levels (merged) ->     2        4        6        8       10
#   ml2 (mergeLevels=2)  eff Pgm steps ~  4        8       12       16       20
#   ml4 (mergeLevels=4)  eff Pgm steps ~  8       16       24       32       40
#
# The coarsest grid is reached once the row count drops below min_coarse_rows (64), so beyond some
# max_levels the extra allowance is inert -- the point of THIS sweep is to find, for each merge factor,
# the max_levels KNEE where the hierarchy stops deepening and to see whether a merge factor paired with
# a DELIBERATELY CAPPED depth (fewer, coarser levels AND fewer of them) finally beats plain Pgm on
# s/step. i.e. does mergeLevels only pay off when you ALSO cap max_levels to shed the coarse tail?
#
# PROTOTYPE SCOPE: the LOCALIZED MG path (Schwarz{Multigrid} on a per-rank local Csr), same as 3c.
# Configs are generated on the fly from the committed merge2/merge4 base configs by rewriting the
# single "max_levels": N entry; the merged coarsener (neon::pgmMerge2 / neon::pgmMerge4) is untouched.
#
# All CACHED (cacheSolver=true, preconditionerRebuildInterval=0), matching Phase 3c: MergedPgm is
# UpdateMatrixValue, so the composed prolongation is built once and only the merged coarse operator's
# values refresh per solve.
#
# What to read (SUMMARY table):
#   s/step        does a capped-depth merge finally beat plain Pgm (Phase-3c ml1 ~= 3.10 s/step)?
#   p_iters/it    convergence vs depth -- expect it to WORSEN as max_levels shrinks (weaker coarse solve).
#   cont          continuity error -- must not degrade.
# Cross-check #levels in each setup log to confirm the hierarchy actually got capped where intended.
#
# Requires the Phase-0 restart (processor*/$RESTART/); aborts loudly if missing.

STUDY_TYPE=mergelevels-maxlevels
STEPS="${STEPS:-50}"
# sbatch copies the script into a spool dir, so $0/BASH_SOURCE point there, not at the
# real script. Fall back to $PWD (guaranteed correct by #SBATCH --chdir) when the sibling
# common script isn't next to us.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"

require_restart

# Reused temp config generated per variant. Lives under system/gko/ so its path is valid as a
# case-relative configFile (the same convention the committed merge configs use). Cleaned at exit.
# NB: NOT a dotfile -- a leading-dot path component (system/gko/.foo) makes OpenFOAM's dictionary
# tokenizer reject the configFile entry ("ill defined primitiveEntry"). Keep the basename plain.
SWEEP_CFG="system/gko/mergesweep-papertmp.json"
cleanup_sweep_cfg() { rm -f "$SWEEP_CFG"; }
trap cleanup_sweep_cfg EXIT   # NB: chains with common.sh's restore() trap (both fire on EXIT).

build_variant_config() {
    # $1 = mergeLevels (2|4), $2 = max_levels. Clone the committed merge<N> base config, rewriting only
    # its "max_levels": N entry. Everything else (mg_level neon::pgmMerge<N>, smoothers, criteria) is
    # inherited verbatim so the sweep isolates the depth cap.
    local merge="$1" maxlevels="$2"
    local base="system/gko/p-multigrid-localized-solver-merge${merge}.json"
    if [ ! -f "$base" ]; then echo "!! missing base config $base" >&2; return 1; fi
    sed -E "s#(\"max_levels\"[[:space:]]*:[[:space:]]*)[0-9]+#\1${maxlevels}#" "$base" > "$SWEEP_CFG" \
        || return 1
    # Guard: fail loudly if the substitution did not take (base schema changed).
    grep -qE "\"max_levels\"[[:space:]]*:[[:space:]]*${maxlevels}\b" "$SWEEP_CFG" \
        || { echo "!! failed to set max_levels=$maxlevels in $base" >&2; return 1; }
}

build_variant_fvsolution() {
    # Point the p configFile at the generated sweep config, CACHED (matches Phase 3c).
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${SWEEP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

run_variant() {
    local name="$1" merge="$2" maxlevels="$3" desc="$4"
    build_variant_config "$merge" "$maxlevels"   || { echo "   skip $name"; return; }
    build_variant_fvsolution                     || { echo "   skip $name"; return; }
    run_one "$name" "$TMP_FVSOL" "$desc"
}

# Build the 2 x 5 variant grid: mergeLevels in {2,4} x max_levels in {2,4,6,8,10}.
declare -A VARIANT
ORDER=()
for merge in 2 4; do
    for L in 2 4 6 8 10; do
        name="ml${merge}_L${L}"
        VARIANT[$name]="$merge $L | localized MG, CACHED, MergedPgm mergeLevels=$merge, max_levels=$L (eff ~$((merge*L)) Pgm steps)"
        ORDER+=("$name")
    done
done

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "Phase 3d: $STEPS iterations/variant from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    params="${spec%% |*}"; desc="${spec#*| }"
    read -r merge maxlevels _ <<< "$params"
    run_variant "$name" "$merge" "$maxlevels" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   compare s/step vs plain-Pgm baseline (Phase 3c ml1),"
echo "and watch p_iters climb as max_levels shrinks -- find each merge factor's depth knee."
