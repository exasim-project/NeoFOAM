#!/bin/bash
#SBATCH --job-name=phase0-paper-spinup
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase0-paper-study-spinup.sh
# Or run live: salloc -w gpu-nvidia-h200-3 -n32 -t 08:00:00 --gres gpu:4   then   ./phase0-paper-study-spinup.sh
# (the #SBATCH lines are inert comments when run interactively; override any at submit, e.g.
#  sbatch -w gpu-nvidia-h200-5 -t 04:00:00 phase0-paper-study-spinup.sh)
#
# Optimization-paper study -- PHASE 0: restart generation (spin-up).
#
# Marches the case ONCE from the uniform 0/ field to iteration $RESTART (1000) with the REFERENCE
# solver (fp64 PCG + global Multigrid, system/gko/p-multigrid.json) and freezes processor*/$RESTART/
# as the semi-converged restart every later phase starts from.
#
# Why the reference (fp64, unoptimized) solver: the restart must be a clean, SOLVER-AGNOSTIC flow
# field. Spinning up with a heavily-optimized solver (mixed precision, aggressive cache intervals)
# would bake solver-specific error into the ground-truth restart and contaminate every downstream
# comparison. The restart iteration is FIXED at 1000 -- not a swept parameter.
#
# Output: processor*/$RESTART/ (the restart) + paperParamStudyResults/spinup/<name>-<ts>.log with the
# full residual / force-coefficient history for the convergence-justification plot.
#
# Usage:   ./phase0-paper-study-spinup.sh          # generate the restart (skips if it already exists)
#          FORCE=1 ./phase0-paper-study-spinup.sh  # regenerate from scratch even if $RESTART/ exists
#
# Idempotent: the artifact is processor*/$RESTART/, so a re-run with the restart already present is a
# no-op unless FORCE=1. NB the guard is the restart DIRECTORY, not the log (delete the dir to redo).

STUDY_TYPE=spinup
# The restart does not exist yet, so DON'T let the common file auto-pin the RESTART->RESTART+STEPS
# measured window; we pin our own 0 -> RESTART spin-up window below.
PAPER_SKIP_WINDOW_PIN=1
# sbatch copies the script into a spool dir, so $0/BASH_SOURCE point there, not at the
# real script. Fall back to $PWD (guaranteed correct by #SBATCH --chdir) when the sibling
# common script isn't next to us.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"

# Spin-up pressure solver: the REFERENCE fp64 Cg + global Pgm Multigrid (p-multigrid.json), same as
# Phase 1/2, so the restart is produced by the exact solver the study measures.
#
# History: cold-start MG spin-ups used to abort NON-DETERMINISTICALLY with gko::DimensionMismatch in
# MultigridState::run_mg_cycle around step 30-140. Root cause was NeoN's Strategy-3 scratch-Workspace
# reuse being wrongly applied to updatable multigrid regenerated with caching off: the reused
# Workspace, sized for a previous hierarchy's coarse levels, mismatched the freshly built coarse
# operators once the value-dependent Pgm aggregation shifted (the matrix varies between solves --
# non-deterministic GPU-atomic assembly, strongest during the transient). FIXED in NeoN
# (ginkgoDistributed.cpp: only non-updatable solvers reuse the Workspace; see
# [[neon-cachesolver-not-a-bug-corrupted-build]]). Verified: a full 1000-step MG cold-start spin-up
# now runs with zero DimensionMismatch. Fall back to SPINUP_CONFIG=p-cg.json (plain PCG, no multigrid)
# only if a regression reappears.
SPINUP_CONFIG="${SPINUP_CONFIG:-p-multigrid.json}"
NAME="spinup-${RESTART}it"
DESC="spin-up 0 -> $RESTART with reference $SPINUP_CONFIG (fp64 Cg + global Multigrid), no opt"

REF_CONFIG="$SPINUP_CONFIG"   # build_reference_fvsolution reads REF_CONFIG
REF_FVSOL="$(build_reference_fvsolution)" \
    || { echo "!! could not build spin-up fvSolution ($SPINUP_CONFIG)"; exit 1; }

pin_window 0 "$RESTART"

if [ -d "processor0/$RESTART" ] && [ -z "${FORCE:-}" ]; then
    echo "================================================================"
    echo " restart processor*/$RESTART already exists -- skipping spin-up"
    echo "   (set FORCE=1 to regenerate, or delete processor*/$RESTART/ first)"
    echo "================================================================"
else
    # Fresh start: drop every written time except 0/ so the spin-up is reproducible (also clears any
    # stale $RESTART/ on a FORCE regeneration). run_one always writes a new log here.
    reset_times 0
    FORCE=1 run_one "$NAME" "$REF_FVSOL" "$DESC"
fi

print_summary

echo
echo "restart field: processor*/$RESTART/  (start point for Phase 1 cost break-down and all sweeps)"
echo "residual / force history: $RESULTS/${NAME}-*.log"
echo "Next: ./phase1-paper-study-costbreakdown.sh"
