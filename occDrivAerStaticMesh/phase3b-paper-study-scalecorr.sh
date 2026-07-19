#!/bin/bash
#SBATCH --job-name=phase3b-paper-scalecorr
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase3b-paper-study-scalecorr.sh
# Or run live: salloc ... then ./phase3b-paper-study-scalecorr.sh
# Override:    STEPS=250 ; run a subset by name, e.g. ./phase3b-...sh cgmg-sc-on mg-sc-on
#
# Optimization-paper study -- PHASE 3b: impact of MULTIGRID SCALE CORRECTION, on the CACHED configs.
#
# Phase 3a established that solver-hierarchy caching (cacheSolver=true, update_matrix_value) is a
# safe, free win (-29.6% s/step at identical convergence) for Cg+global-Multigrid. Building on that,
# this sweep isolates the *scale-correction* knob (Braess/Rayleigh scaling of the smoother sweeps)
# with everything else -- caching, restart, window -- held fixed. A 3x2 (scale-correction setting x
# solver form), 50 SIMPLE iterations per variant:
#
#                          OFF (pre=none)      FWD-only (pre=forward)   ON (pre=forward/post=backward)
#   Cg + global Multigrid  cgmg-sc-off         cgmg-sc-fwd              cgmg-sc-on
#   Multigrid as SOLVER    mg-sc-off           mg-sc-fwd                mg-sc-on
#
# SCALE-CORRECTION RECIPE. Correction is applied PER SMOOTHER via the Ir smoother's scale_correction
# mode (gko::solver::scale_correction_mode), with DISTINCT pre/post smoothers (post_uses_pre=false):
#   "forward"  : solve delta = omega*M^-1 r, then Rayleigh-correct delta.
#   "backward" : Rayleigh-correct r as initial guess, then apply solver (matches OpenFOAM
#                GAMGSolver::scale()).
# The FWD-only variant sets pre="forward", post="none" (correction on the down-sweep only); the ON
# variant sets pre="forward", post="backward" (symmetric). This is NOT the MG-level
# `scale_correction: true` boolean (a separate coarse-correction scaling in multigrid.cpp) and NOT an
# outer-Ir wrapper -- both left OFF. Each FWD/ON config differs from its OFF base in EXACTLY the
# changed smoother mode(s) (verified by diff), so the grid isolates scale correction.
#
# The OFF column (p-multigrid.json / p-multigrid-solver.json, cached) reproduces the Phase-3a cached
# variants (cgmg-cached / mgsolver-cached) as a consistency check (note: Phase 3a ran 250 steps;
# here every variant runs 50).
#
# ALL variants CACHED (cacheSolver=true, preconditionerRebuildInterval=0): the scale-corrected
# Multigrid stays updatable, so it caches via update_matrix_value exactly like the plain hierarchy.
#
# NOTE the prior localized dead-end (memory: scalecorr + LOCALIZED Schwarz{MG} = 9x slower, per-
# subdomain Rayleigh scaling inconsistent across the decomposition). These are GLOBAL MG (Cg/MG
# outer over the whole distributed matrix, Schwarz only inside the smoother), NOT localized, so that
# dead-end should not apply -- this run tests exactly that.
#
# What to read:
#   s/step        does scale correction pay for itself? (extra work per sweep vs fewer iterations)
#   p_iters/it    the point of scale correction -- it should REDUCE the pressure iteration count if
#                 the smoother was mis-scaled; watch whether it helps Cg+MG (already good at ~18)
#                 or mainly rescues the MG-as-solver form (52-77 iters in Phase 3a).
#   cont          continuity error -- must not degrade (the localized dead-end showed up here).
#
# Requires the Phase-0 restart (processor*/$RESTART/); aborts loudly if missing.

STUDY_TYPE=scalecorr
STEPS="${STEPS:-50}"            # 50 SIMPLE iterations per variant (set before sourcing: pins the window)
# sbatch copies the script into a spool dir, so $0/BASH_SOURCE point there, not at the
# real script. Fall back to $PWD (guaranteed correct by #SBATCH --chdir) when the sibling
# common script isn't next to us.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"

require_restart

build_variant_fvsolution() {
    # $1 = config json basename (cacheSolver+interval injected -> always cached here)
    local cfg="$1" cache="true" interval="0"
    if [ ! -f "system/gko/$cfg" ]; then echo "!! missing system/gko/$cfg" >&2; return 1; fi
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$cfg;\n        cacheSolver      $cache;\n        preconditionerRebuildInterval $interval;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

run_variant() {
    build_variant_fvsolution "$2" || { echo "   skip $1"; return; }
    run_one "$1" "$TMP_FVSOL" "$3"
}

# name -> "config | desc". Three scale-correction settings x two solver forms. The ON/FWD configs
# differ from their OFF base in ONLY the smoother scale_correction mode(s) (verified by diff):
#   off  : pre=none,    post=none
#   fwd  : pre=forward, post=none     (forward correction on the down-sweep only)
#   on   : pre=forward, post=backward (symmetric; backward matches OpenFOAM GAMGSolver::scale())
declare -A VARIANT=(
    [cgmg-sc-off]="p-multigrid.json | Cg + global Multigrid, CACHED, scale correction OFF (pre=none, post=none)"
    [cgmg-sc-fwd]="p-multigrid-scfwd.json | Cg + global Multigrid, CACHED, scale correction pre=forward only (post=none)"
    [cgmg-sc-on]="p-multigrid-sc.json | Cg + global Multigrid, CACHED, scale correction pre=forward, post=backward"
    [mg-sc-off]="p-multigrid-solver.json | Multigrid as SOLVER, CACHED, scale correction OFF (pre=none, post=none)"
    [mg-sc-fwd]="p-multigrid-solver-scfwd.json | Multigrid as SOLVER, CACHED, scale correction pre=forward only (post=none)"
    [mg-sc-on]="p-multigrid-solver-sc.json | Multigrid as SOLVER, CACHED, scale correction pre=forward, post=backward"
)
# Default: the full 3x2 fresh at $STEPS steps (self-contained; the OFF cells reproduce the Phase-3a
# cached configs as a consistency check). Pass names explicitly to run a subset.
ORDER=(cgmg-sc-off cgmg-sc-fwd cgmg-sc-on mg-sc-off mg-sc-fwd mg-sc-on)

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "Phase 3b: $STEPS iterations/variant from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    cfg="${spec%% *}"; desc="${spec#*| }"
    run_variant "$name" "$cfg" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   compare s/step (speed) AND p_iters/cont (does scale correction help?)"
