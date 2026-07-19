#!/bin/bash
#SBATCH --job-name=phase3h-paper-mgsc-coarse-reltol
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase3h-paper-study-mgsc-coarse-reltol.sh
# Or run live: ./phase3h-paper-study-mgsc-coarse-reltol.sh            (subset: ./...sh L5e2 L4e3)
# Override:    STEPS=50   LEVELS="3 4 5 6"   TOLS="0.1 0.01 0.001 0.0001"
#
# PHASE 3h: 2-D sweep (max_levels x coarse relative-tolerance) for the GLOBAL scale-corrected MG.
# Base = p-multigrid-mgsc-merge2-lcg.json (the §4.11j best building blocks):
#   solver::Cg + Multigrid(scale_correction=true) + pgmMerge2 + localized-CG coarse
#   (coarsest_solver = Schwarz{ Cg + Jacobi }).
#
# §4.11j/coarse-iters (phase3g) used a FIXED coarse max_iters (no early exit): the coarse Cg always burned
# the full count, and the curve was monotone (more coarse iters -> fewer outer iters, best at c10). A fixed
# count over/under-solves depending on depth. This sweep replaces the coarse Iteration criterion with a
# RELATIVE-RESIDUAL criterion (ResidualNorm, baseline=initial_resnorm, i.e. ||r_k||/||r_0|| < tol -- the
# classic Krylov relative tolerance; since the coarse solve starts from a zero guess, initial_resnorm ==
# rhs_norm here), so the coarse Cg adapts its iteration count to what each level actually needs.
#   + an Iteration(50) SAFETY CAP so a tight tol can't run away (Ginkgo stops on OR of criteria).
#
# 2-D grid: max_levels in LEVELS x rel-tol in TOLS. Variant name = L{lvl}e{n} where tol = 10^-n
#   (L5e2 = max_levels 5, coarse rel-tol 1e-2). Defaults: LEVELS="3 4 5 6", TOLS="0.1 0.01 0.001 0.0001".
# jq sets ONLY .preconditioner.max_levels and .coarsest_solver.local_solver.criteria (outer 150 + smoother
# 1 untouched). All CACHED (cacheSolver=true, interval=0).
#
# Read (SUMMARY per cell): p_iters (outer CG V-cycles), p_ms/solve, s/step, cont (~3e-6). The question:
# does an adaptive (tolerance) coarse solve beat the best fixed-count cell (§4.11j L5, c10 = 7.5 iters,
# 3.660 s/step), and where in (levels, tol) is the joint optimum?
#
# Requires the Phase-0 restart (processor*/$RESTART/); aborts loudly if missing.

# SC=true|false -- MG-level scale correction. The sc=true grid (§4.11k) found L4 x tol 0.1 = 3.620,
# but a clean sc=ON/OFF control at that very cell found sc is a NET LOSS in this config
# (3.640 ON vs 3.360 OFF): sc cuts outer iters 13.8->7.7 (1.79x) but MORE THAN DOUBLES per-V-cycle
# communication (85.65 -> 192.56 ms/V-cycle, measured -- analyze-comm-attribution.py), and this solve
# is sync-bound. §4.11b's "sc wins" predates merge2 + localized coarse + the rel-tol criterion, all of
# which made the baseline cycle cheaper and shrank sc's iteration edge (16.5->6 = 2.75x back then).
# So the whole grid must be re-run with sc=false: the (levels, tol) optimum may sit elsewhere without it.
# Results go to a SEPARATE dir per SC so the two grids don't collide.
#
# SC=post -- scale correction with ONLY the post-smooth pass (§4.12.4). The pre pass was measured to
# deliver ZERO convergence benefit (14.0 iters vs 13.8 for no sc at all) while costing real comm, so
# `both` is never the right way to run sc; post-only is the mechanism minus the waste. It should sit
# BETWEEN the two branches (at L4/tol0.1: none 13.8 iters, post 9.3, both 7.6), and because it retains
# partial scaling repair its DEPTH optimum may differ from either (sc=ON: L4, sc=OFF: L3) -- hence its
# own grid rather than an inference. Requires the NEON_MGSC_MODE probe build (multigrid.cpp, not for
# merge); the env var is forwarded to all ranks via MPIRUN_FORWARD_ENV.
SC="${SC:-true}"
case "$SC" in
    true)  STUDY_TYPE=mgsc-coarse-reltol   ;;   # stock: both passes
    false) STUDY_TYPE=mgnosc-coarse-reltol ;;   # sc off entirely
    post)  STUDY_TYPE=mgscpost-coarse-reltol
           export NEON_MGSC_MODE=post
           export MPIRUN_FORWARD_ENV="${MPIRUN_FORWARD_ENV:-} NEON_MGSC_MODE" ;;
    *) echo "!! SC must be true|false|post (got '$SC')" >&2; exit 1 ;;
esac
STUDY_TYPE="${STUDY_TYPE}${STUDY_SUFFIX:-}"   # e.g. -merge1 for the merge-levels sweep
STEPS="${STEPS:-50}"
LEVELS="${LEVELS:-3 4 5 6}"
TOLS="${TOLS:-0.1 0.01 0.001 0.0001}"
COARSE_CAP="${COARSE_CAP:-50}"     # Iteration safety cap on the coarse solve
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

BASE_CFG="system/gko/p-multigrid-mgsc-merge2-lcg.json"   # global mgsc + pgmMerge2 + localized-CG coarse
SWEEP_CFG="system/gko/mgscreltol-papertmp.json"
cleanup_sweep_cfg() { rm -f "$SWEEP_CFG"; }
trap cleanup_sweep_cfg EXIT   # chains with common.sh's restore() trap

# tol -> exponent code n (10^-n) for the variant name; e.g. 0.001 -> 3
tol_code() {
    case "$1" in
        0.1)    echo 1 ;;
        0.01)   echo 2 ;;
        0.001)  echo 3 ;;
        0.0001) echo 4 ;;
        # loose-end probe (§4.11k): 1e-1 won every row, so test even looser coarse tolerances.
        # Named by their digits (0.15 -> e015) -- no dots in log names (the harness globs on them).
        0.15)   echo 015 ;;
        0.2)    echo 02 ;;
        0.25)   echo 025 ;;
        *)      echo "$1" | sed -E 's/[^0-9-]//g' ;;   # fallback: strip to digits
    esac
}

build_variant_config() {
    # $1 = max_levels, $2 = rel-tol. Clone base; pin max_levels; swap coarse criteria to ResidualNorm+cap.
    local ml="$1" tol="$2"
    if [ ! -f "$BASE_CFG" ]; then echo "!! missing base config $BASE_CFG" >&2; return 1; fi
    # SC=post keeps scale_correction=true in the config; the pass split is done by NEON_MGSC_MODE.
    local sc_json="$SC"; [ "$SC" = "post" ] && sc_json=true
    # MERGE (default 2) selects the pgmMerge{N} coarsener; N=1 = plain Pgm.
    local merge_name="neon::pgmMerge${MERGE:-2}"
    jq --argjson ml "$ml" --argjson tol "$tol" --argjson cap "$COARSE_CAP" --argjson sc "$sc_json" \
       --arg mg "$merge_name" '
        .preconditioner.max_levels = $ml
        | .preconditioner.scale_correction = $sc
        | .preconditioner.mg_level = [$mg]
        | .preconditioner.coarsest_solver.local_solver.criteria = [
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":$tol},
            {"type":"Iteration","max_iters":$cap}
          ]' "$BASE_CFG" > "$SWEEP_CFG" || return 1
    local got
    got=$(jq '.preconditioner.coarsest_solver.local_solver.criteria[0].reduction_factor' "$SWEEP_CFG")
    [ "$got" = "$tol" ] || { echo "!! failed to set coarse rel-tol=$tol (got $got)" >&2; return 1; }
    got=$(jq '.preconditioner.scale_correction' "$SWEEP_CFG")
    [ "$got" = "$sc_json" ] || { echo "!! failed to set scale_correction=$sc_json (got $got)" >&2; return 1; }
}

build_variant_fvsolution() {
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${SWEEP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

run_variant() {
    local name="$1" ml="$2" tol="$3" desc="$4"
    build_variant_config "$ml" "$tol" || { echo "   skip $name"; return; }
    build_variant_fvsolution          || { echo "   skip $name"; return; }
    run_one "$name" "$TMP_FVSOL" "$desc"
}

declare -A VARIANT
ORDER=()
for L in $LEVELS; do
    for T in $TOLS; do
        n=$(tol_code "$T")
        name="L${L}e${n}"
        VARIANT[$name]="$L $T | GLOBAL Cg+MG mgsc, CACHED, pgmMerge2, localized-CG coarse rel-tol=$T (cap $COARSE_CAP), max_levels=$L"
        ORDER+=("$name")
    done
done

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "Phase 3h (global mgsc + merge2 + localized-CG coarse; 2-D max_levels x rel-tol): $STEPS iters/variant from restart $RESTART"
echo "  LEVELS={$LEVELS}  TOLS={$TOLS}  coarse-cap=$COARSE_CAP  variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    ml="${spec%% *}"; rest="${spec#* }"; tol="${rest%% |*}"; desc="${spec#*| }"
    run_variant "$name" "$ml" "$tol" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   2-D knee: does adaptive coarse tol beat the fixed-count best?"
echo "  reference: §4.11j L5 (fixed coarse max_iters=10) = 7.5 outer iters, 3.660 s/step"
echo "phase3h done"
