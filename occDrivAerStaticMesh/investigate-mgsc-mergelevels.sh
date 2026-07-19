#!/bin/bash
#SBATCH --job-name=investigate-mgsc-mergelevels
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Does MG-LEVEL scale correction compound with mergeLevels (global, distributed MergedPgm)?
#
# Two established single-lever wins/nulls (same 07-13 build, same-session, cached, 50 steps):
#   * mergeLevels (global): ~wash vs plain Pgm (fewer levels -> cheaper cycle ~cancels more iters).
#   * MG-level scale_correction=true (mgsc): pressure iters 16.5->6.0, s/step ~4.5->4.3, works under Cg.
# Hypothesis: fewer LEVELS (merge) x fewer ITERATIONS (mgsc) could compound. A 2x3 grid, same session:
#
#             merge1 (plain Pgm)        merge2 (pgmMerge2)          merge3 (pgmMerge3)
#   no-sc     p-multigrid.json          p-multigrid-merge2.json     p-multigrid-merge3.json
#   mgsc      p-multigrid-mgsc.json     p-multigrid-mgsc-merge2.json p-multigrid-mgsc-merge3.json
#
# All Cg + global MG, CACHED (cacheSolver=true, interval=0). mgsc is MG-level scale_correction=true
# (Rayleigh scaling at the restrict/prolong boundary); it drives MergedPgm's composed prolong/coarse
# ops exactly like plain Pgm, so this also confirms mgsc x distributed-MergedPgm compose correctly.
#
# Read: p_iters (does merge+mgsc stay ~6, or does coarser transfer inflate it?), p_ms/solve, s/step
# (best cell wins), cont (must stay ~3e-6; no divergence). Single-run variance is ~20% -- trust the
# p_iters signal over small s/step gaps.

STUDY_TYPE=mgsc-mergelevels
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

declare -A VARIANT=(
    [nosc-m1]="p-multigrid.json | Cg+MG, CACHED, no sc, plain Pgm (merge1)"
    [nosc-m2]="p-multigrid-merge2.json | Cg+MG, CACHED, no sc, pgmMerge2"
    [nosc-m3]="p-multigrid-merge3.json | Cg+MG, CACHED, no sc, pgmMerge3"
    [mgsc-m1]="p-multigrid-mgsc.json | Cg+MG, CACHED, MG-level sc, plain Pgm (merge1)"
    [mgsc-m2]="p-multigrid-mgsc-merge2.json | Cg+MG, CACHED, MG-level sc, pgmMerge2"
    [mgsc-m3]="p-multigrid-mgsc-merge3.json | Cg+MG, CACHED, MG-level sc, pgmMerge3"
)
ORDER=(nosc-m1 nosc-m2 nosc-m3 mgsc-m1 mgsc-m2 mgsc-m3)

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "mgsc x mergeLevels: $STEPS iterations/variant from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    cfg="${spec%% *}"; desc="${spec#*| }"
    run_variant "$name" "$cfg" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   does merge (fewer levels) x mgsc (fewer iters) compound?"
