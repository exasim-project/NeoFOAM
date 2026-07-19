#!/bin/bash
#SBATCH --job-name=phase3-paper-cache-compare
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase3-paper-study-cache-compare.sh
# Or run live: salloc -w gpu-nvidia-h200-3 -n32 -t 12:00:00 --gres gpu:4  then  ./phase3-paper-study-cache-compare.sh
# Override:    STEPS=250 (iterations) ; run a subset by name, e.g. ./phase3-...sh cgmg-cached mgsolver-cached
#
# Optimization-paper study -- PHASE 3a: solver-hierarchy REUSE (update_matrix_value) vs REBUILD.
#
# A 2x2 comparison, each run 250 SIMPLE iterations from the iteration-$RESTART restart:
#
#                         no caching (rebuild hierarchy every solve)   cached (build once, update in place, NEVER rebuild)
#   Cg + global Multigrid   cgmg-nocache                                cgmg-cached
#   Multigrid as SOLVER     mgsolver-nocache                            mgsolver-cached
#   (no outer Krylov)
#
# "Cached" = cacheSolver=true + preconditionerRebuildInterval=0 -> Strategy 1b builds the Pgm
# hierarchy once (solve 1) and thereafter refreshes only the Galerkin coarse-operator VALUES via
# gko::UpdateMatrixValue::update_matrix_value (Multigrid + Pgm both implement it), reusing the
# aggregation. NEVER a forced full rebuild (interval 0). "no caching" (cacheSolver=false) regenerates
# the whole hierarchy -- re-running the Pgm aggregation -- on every solve (the Phase-1 break-down's
# 26.8% ginkgo.solverSetup cost). Both no-cache paths are safe on this build (workspace-reuse fix).
#
# What to read:
#   s/step        speed -- the point of caching (should drop most for the cached variants).
#   p_iters/it    convergence quality -- a stale reused hierarchy can need MORE V-cycles/Cg iters as
#                 the operator drifts from the solve-1 aggregation. MG-as-solver is LESS forgiving of
#                 drift than MG-as-preconditioner (no outer Krylov to correct), so watch this closely.
#   cont          continuity error -- the solution must not degrade.
#   CACHE STATUS  confirms the cached variants build once then reuse (never rebuild).
#
# Requires the Phase-0 restart (processor*/$RESTART/); aborts loudly if missing.

STUDY_TYPE=cache-compare
STEPS=250                       # 250 SIMPLE iterations per variant (set before sourcing: pins the window)
# sbatch copies the script into a spool dir, so $0/BASH_SOURCE point there, not at the
# real script. Fall back to $PWD (guaranteed correct by #SBATCH --chdir) when the sibling
# common script isn't next to us.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"

require_restart

build_variant_fvsolution() {
    # $1 = config json basename   $2 = cacheSolver (true|false)   $3 = preconditionerRebuildInterval
    # Swap the p configFile + inject the caching keys into the case3 template (sed, not foamDictionary,
    # which retokenises the path). U/k/omega blocks are the fixed baseline, untouched.
    local cfg="$1" cache="$2" interval="$3"
    if [ ! -f "system/gko/$cfg" ]; then echo "!! missing system/gko/$cfg" >&2; return 1; fi
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$cfg;\n        cacheSolver      $cache;\n        preconditionerRebuildInterval $interval;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

run_variant() {
    # $1 name   $2 config json   $3 cacheSolver   $4 rebuildInterval   $5 desc
    build_variant_fvsolution "$2" "$3" "$4" || { echo "   skip $1"; return; }
    run_one "$1" "$TMP_FVSOL" "$5"
}

# name -> "config cacheSolver interval desc"
declare -A VARIANT=(
    [cgmg-nocache]="p-multigrid.json false 0 | Cg + global Multigrid, NO caching (Pgm hierarchy rebuilt every solve)"
    [cgmg-cached]="p-multigrid.json true 0 | Cg + global Multigrid, CACHED (update_matrix_value, never rebuild)"
    [mgsolver-nocache]="p-multigrid-solver.json false 0 | Multigrid as SOLVER (no outer Krylov), NO caching"
    [mgsolver-cached]="p-multigrid-solver.json true 0 | Multigrid as SOLVER (no outer Krylov), CACHED (update_matrix_value, never rebuild)"
)
ORDER=(cgmg-nocache cgmg-cached mgsolver-nocache mgsolver-cached)

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "Phase 3a: $STEPS iterations/variant from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    cfg="${spec%% *}"; rest="${spec#* }"; cache="${rest%% *}"; rest="${rest#* }"
    interval="${rest%% *}"; desc="${spec#*| }"
    run_variant "$name" "$cfg" "$cache" "$interval" "$desc"
done

print_summary

# ------------------------------------------------ cache status: rebuild vs in-place update per variant
report_cache_status() {
    echo
    echo "###################### CACHE STATUS (per variant) ######################"
    printf "%-20s %14s %14s   %s\n" run "MG rebuilds" "MG updates" "verdict"
    local c log rb up
    for c in "${RUNS[@]}"; do
        log="${RUN_LOG[$c]}"; [ -n "$log" ] && [ -f "$log" ] || continue
        # "reuse(update_matrix_value)" comes only from an UPDATABLE solver (the cached p multigrid /
        # Phi); "rebuild(generate)" is a from-scratch (re)generate. The diagnostic prints only when
        # cacheSolver_=true, so no-cache variants show 0/0 for p (p regenerates silently every solve).
        rb=$(grep -c 'p-cache: rebuild(generate) ' "$log" 2>/dev/null); rb=${rb:-0}
        up=$(grep -c 'p-cache: reuse(update_matrix_value)' "$log" 2>/dev/null); up=${up:-0}
        local verdict
        case "$c" in
            *cached*)  verdict=$([ "$up" -gt 0 ] && echo "cached: built once, updated in place (no rebuild)" || echo "!! caching did NOT engage");;
            *nocache*) verdict="no cache: hierarchy regenerated every solve";;
            *)         verdict="";;
        esac
        printf "%-20s %14s %14s   %s\n" "$c" "$rb" "$up" "$verdict"
    done
    echo "########################################################################"
    echo "(p-cache diagnostic prints only for cacheSolver=true solves; k/omega add a constant offset)"
}
report_cache_status

echo
echo "logs under: $RESULTS/   |   compare s/step (speed) AND p_iters/cont (convergence quality)"
