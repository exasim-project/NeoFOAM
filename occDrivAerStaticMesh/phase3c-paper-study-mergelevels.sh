#!/bin/bash
#SBATCH --job-name=phase3c-paper-mergelevels
#SBATCH --nodelist=gpu-nvidia-h200-2
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase3c-paper-study-mergelevels.sh
# Or run live: salloc ... then ./phase3c-paper-study-mergelevels.sh
# Override:    STEPS=50 ; run a subset by name, e.g. ./phase3c-...sh gm1 gm2 gm3 gm4   (global sweep)
#                                              or   ./phase3c-...sh gm1 gm1-sc         (scaleCorr A/B)
#
# Optimization-paper study -- PHASE 3c: impact of mergeLevels (MergedPgm SpGEMM coarsening).
#
# OpenFOAM GAMG's `mergeLevels N` collapses N agglomeration steps into one coarser level, so the
# V-cycle sweeps FEWER, coarser levels. NeoN's Pgm has no equivalent; MergedPgm (NeoN Ginkgo
# extension) realises it by running Pgm N times internally and SpGEMM-composing the piecewise-constant
# prolongations into a single merged multigrid level (see MergeLevelsStudyPlan-2026-07-09.md). This is
# the best-motivated lever after caching: the Phase-1 nsys break-down showed the solve is
# DISPATCH-bound, and the deep coarse levels do microseconds of arithmetic yet each still pays a full
# kernel launch + stream sync -- halving the level count removes that tail.
#
# The merged coarsener is named in each config's mg_level: "mg_level": ["neon::pgmMerge<N>"]
# (registered in the GinkgoSolver ctor). MergedPgm dispatches on the fine-op type, so the SAME named
# coarsener drives both scopes below.
#
# TWO SCOPES:
#   (A) LOCALIZED MG (Schwarz{Multigrid} on a per-rank local Csr) -- the original prototype. Coarsening
#       is a plain local Csr, no distributed bookkeeping. Prior result: no speedup (dispatch saving
#       ~cancels a +51% iter cost; ml1 best). Kept here as the localized reference.
#           ml1  p-multigrid-localized-solver.json          plain Pgm        (baseline, ~8 levels)
#           ml2  p-multigrid-localized-solver-merge2.json   neon::pgmMerge2  (~4 levels)
#           ml3  p-multigrid-localized-solver-merge3.json   neon::pgmMerge3  (~3 levels)
#           ml4  p-multigrid-localized-solver-merge4.json   neon::pgmMerge4  (~2 levels)
#
#   (B) GLOBAL MG (Cg + global Multigrid over the whole distributed matrix; Schwarz only inside the
#       smoother) -- the distributed MergedPgm branch (generateDistributed: block-diagonal prolong
#       composed rank-locally, A_merged = last inner Pgm's distributed coarse op). Unlike (A), the
#       merged coarse op KEEPS cross-rank coupling, so the convergence/dispatch trade may differ from
#       the localized null result. Run WITH and WITHOUT multigrid scale correction (per Phase-3b):
#           scaleCorr OFF (pre=none, post=none)          scaleCorr ON (pre=forward, post=backward)
#           gm1  p-multigrid.json          plain Pgm      gm1-sc  p-multigrid-sc.json
#           gm2  p-multigrid-merge2.json   pgmMerge2      gm2-sc  p-multigrid-sc-merge2.json
#           gm3  p-multigrid-merge3.json   pgmMerge3      gm3-sc  p-multigrid-sc-merge3.json
#           gm4  p-multigrid-merge4.json   pgmMerge4      gm4-sc  p-multigrid-sc-merge4.json
#       The gm*-sc configs differ from gm* ONLY in the two smoother scale_correction modes (none ->
#       forward/backward); mg_level is identical within each merge level. gm1/gm1-sc are the plain-Pgm
#       global baselines (== Phase-3b cgmg-sc-off / cgmg-sc-on).
#
# CAVEAT (scaleCorr x mergeLevels under Cg): scale correction makes the MG preconditioner NONLINEAR,
# which can stall/diverge a plain Cg outer (flexible/FCG is the clean fix). The gm*-sc column tests
# this empirically on top of merging; watch `cont` and p_iters for divergence and treat a blow-up as
# the expected Cg-incompatibility, not a MergedPgm bug.
#
# All CACHED (cacheSolver=true, preconditionerRebuildInterval=0). MergedPgm is UpdateMatrixValue, so
# the composed prolongation is built once and only the merged coarse operator's values refresh per
# solve -- caching (Phase-3a) still engages.
#
# What to read:
#   s/step        does fewer levels beat the (mild) convergence cost of coarser transfer?
#   p_iters/it    coarser interpolation may raise the iteration count -- find the knee.
#   cont          continuity error -- must not degrade (esp. the gm*-sc column, see caveat).
#   #levels       (setup log) confirm the hierarchy actually got shallower.
#
# Requires the Phase-0 restart (processor*/$RESTART/); aborts loudly if missing.

STUDY_TYPE=mergelevels
STEPS="${STEPS:-50}"
# sbatch copies the script into a spool dir, so $0/BASH_SOURCE point there, not at the
# real script. Fall back to $PWD (guaranteed correct by #SBATCH --chdir) when the sibling
# common script isn't next to us.
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

run_variant() {
    build_variant_fvsolution "$2" || { echo "   skip $1"; return; }
    run_one "$1" "$TMP_FVSOL" "$3"
}

declare -A VARIANT=(
    # (A) LOCALIZED MG (per-rank Schwarz{Multigrid} on a local Csr) -- localized reference
    [ml1]="p-multigrid-localized-solver.json | localized MG solver, CACHED, plain Pgm (mergeLevels=1 baseline)"
    [ml2]="p-multigrid-localized-solver-merge2.json | localized MG solver, CACHED, MergedPgm mergeLevels=2"
    [ml3]="p-multigrid-localized-solver-merge3.json | localized MG solver, CACHED, MergedPgm mergeLevels=3"
    [ml4]="p-multigrid-localized-solver-merge4.json | localized MG solver, CACHED, MergedPgm mergeLevels=4"
    # (B) GLOBAL MG (Cg + global Multigrid, distributed MergedPgm), scale correction OFF
    [gm1]="p-multigrid.json | GLOBAL Cg+MG, CACHED, plain Pgm (mergeLevels=1 baseline), scaleCorr OFF"
    [gm2]="p-multigrid-merge2.json | GLOBAL Cg+MG, CACHED, MergedPgm mergeLevels=2, scaleCorr OFF"
    [gm3]="p-multigrid-merge3.json | GLOBAL Cg+MG, CACHED, MergedPgm mergeLevels=3, scaleCorr OFF"
    [gm4]="p-multigrid-merge4.json | GLOBAL Cg+MG, CACHED, MergedPgm mergeLevels=4, scaleCorr OFF"
    # (B) GLOBAL MG, scale correction ON (pre=forward, post=backward) -- see Cg-compat caveat above
    [gm1-sc]="p-multigrid-sc.json | GLOBAL Cg+MG, CACHED, plain Pgm (mergeLevels=1 baseline), scaleCorr ON"
    [gm2-sc]="p-multigrid-sc-merge2.json | GLOBAL Cg+MG, CACHED, MergedPgm mergeLevels=2, scaleCorr ON"
    [gm3-sc]="p-multigrid-sc-merge3.json | GLOBAL Cg+MG, CACHED, MergedPgm mergeLevels=3, scaleCorr ON"
    [gm4-sc]="p-multigrid-sc-merge4.json | GLOBAL Cg+MG, CACHED, MergedPgm mergeLevels=4, scaleCorr ON"
)
ORDER=(ml1 ml2 ml3 ml4 gm1 gm2 gm3 gm4 gm1-sc gm2-sc gm3-sc gm4-sc)

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "Phase 3c: $STEPS iterations/variant from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    cfg="${spec%% *}"; desc="${spec#*| }"
    run_variant "$name" "$cfg" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   compare s/step AND p_iters/cont across mergeLevels 1..4"
echo "  localized: ml1..ml4   |   global: gm1..gm4 (scaleCorr OFF), gm1-sc..gm4-sc (scaleCorr ON)"
