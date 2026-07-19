#!/bin/bash
#SBATCH --job-name=phase3i-paper-localized-reltol
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# PHASE 3i: the LOCALIZED-MG analog of phase3h's 2-D (max_levels x coarse rel-tol) grid.
#
# Architecture (differs from phase3h's GLOBAL MG):
#   solver::Cg  ->  preconditioner::Schwarz  ->  local_solver = solver::Multigrid (per-rank)
# i.e. the ENTIRE multigrid is rank-local; there is no distributed SpMV inside the V-cycle at all,
# only the outer Cg's halo. That is a fundamentally different comm profile from the global MG
# (§4.12.3: sc's A.delta halos dominate there) -- and it is why this branch matters:
#
#   §4.11f measured localized ~3.14 s/step (23.5 iters) vs global mgsc ~3.9 (9 iters) -- LOCALIZED
#   FASTER -- but on the 07-10 build, a CROSS-BUILD comparison that was never re-run same-build.
#   Every §4.12/§4.13 conclusion is about the GLOBAL branch. If localized still wins on the current
#   build, the global-branch optimum is a local optimum of the wrong family.
#
# SC=none|post -- MG-level scale correction on the LOCAL multigrid.
#   The old "sc does not port to localized, 9x slower" dead-end was REFUTED 2026-07-15: that config
#   used an OUTER solver::Ir(scale_correction:"backward") INSTEAD of Cg -- a different (string) knob
#   and a fixed-point outer solver, which alone explains its 446 iters. Cg + Schwarz{MG(sc:true)}
#   converges fine (17.9 iters, cont 2.9e-6).
#   Inside a localized MG the whole hierarchy is rank-local, so sc's A.delta SpMV AND its Rayleigh
#   dots are LOCAL -> sc costs ZERO communication on this branch, only arithmetic.
#   post-only is the mode swept: it is OpenFOAM's default (nPreSweeps_=0 gates the pre pass off) and
#   beats `both` on wall time even under the faithful port (2.401 vs 2.465).
#   Requires the NEON_MGSC_MODE probe build + the faithful-port patch (D^-1 inner op, fused dots).
#
# Sweeps: .preconditioner.local_solver.max_levels  and the local MG's coarsest_solver criteria
# (Ir+Jacobi -> ResidualNorm(tol) + Iteration(cap)), mirroring phase3h's knob on the global branch.
#
# Usage:  LEVELS="4 6 8 10" TOLS="0.25 0.2 0.15 0.1 0.01 0.001 0.0001" ./phase3i-...sh
# Read:   ./peek-grid.py localized-coarse-reltol

SC="${SC:-none}"
case "$SC" in
    none) STUDY_TYPE=localized-coarse-reltol ;;
    post) STUDY_TYPE=localizedsc-coarse-reltol
          export NEON_MGSC_MODE=post
          export MPIRUN_FORWARD_ENV="${MPIRUN_FORWARD_ENV:-} NEON_MGSC_MODE" ;;
    *) echo "!! SC must be none|post (got '$SC')" >&2; exit 1 ;;
esac
STUDY_TYPE="${STUDY_TYPE}${STUDY_SUFFIX:-}"   # e.g. -merge1 for the merge-levels sweep
STEPS="${STEPS:-50}"
# Localized MG coarsens per-rank from a ~49 M-cell local block, so it needs a DEEPER hierarchy than
# the global branch (prior: L10 the sweet spot, depth plateaus past L10) -- hence a deeper range.
LEVELS="${LEVELS:-4 6 8 10}"
TOLS="${TOLS:-0.25 0.2 0.15 0.1 0.01 0.001 0.0001}"
COARSE_CAP="${COARSE_CAP:-50}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

# merge2 to match phase3h's coarsener (the global grids all used pgmMerge2).
BASE_CFG="system/gko/p-multigrid-localized-solver-merge2.json"
SWEEP_CFG="system/gko/locreltol-papertmp.json"
cleanup_sweep_cfg() { rm -f "$SWEEP_CFG"; }
trap cleanup_sweep_cfg EXIT

tol_code() {
    case "$1" in
        0.1) echo 1 ;; 0.01) echo 2 ;; 0.001) echo 3 ;; 0.0001) echo 4 ;;
        0.15) echo 015 ;; 0.2) echo 02 ;; 0.25) echo 025 ;;
        *) echo "$1" | sed -E 's/[^0-9-]//g' ;;
    esac
}

build_variant_config() {
    local ml="$1" tol="$2"
    if [ ! -f "$BASE_CFG" ]; then echo "!! missing base config $BASE_CFG" >&2; return 1; fi
    # NOTE the paths: everything MG lives under .preconditioner.local_solver (inside Schwarz).
    local sc_json=false; [ "$SC" = "post" ] && sc_json=true
    # MERGE (default 2) selects the pgmMerge{N} coarsener on the LOCAL multigrid; N=1 = plain Pgm.
    local merge_name="neon::pgmMerge${MERGE:-2}"
    jq --argjson ml "$ml" --argjson tol "$tol" --argjson cap "$COARSE_CAP" --argjson sc "$sc_json" \
       --arg mg "$merge_name" '
        .preconditioner.local_solver.max_levels = $ml
        | .preconditioner.local_solver.scale_correction = $sc
        | .preconditioner.local_solver.mg_level = [$mg]
        | .preconditioner.local_solver.coarsest_solver.criteria = [
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":$tol},
            {"type":"Iteration","max_iters":$cap}
          ]' "$BASE_CFG" > "$SWEEP_CFG" || return 1
    local got
    got=$(jq '.preconditioner.local_solver.coarsest_solver.criteria[0].reduction_factor' "$SWEEP_CFG")
    [ "$got" = "$tol" ] || { echo "!! failed to set coarse rel-tol=$tol (got $got)" >&2; return 1; }
    got=$(jq '.preconditioner.local_solver.max_levels' "$SWEEP_CFG")
    [ "$got" = "$ml" ] || { echo "!! failed to set max_levels=$ml (got $got)" >&2; return 1; }
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
        name="L${L}e$(tol_code "$T")"
        VARIANT[$name]="$L $T | LOCALIZED Cg+Schwarz{MG}, CACHED, pgmMerge2, local-MG max_levels=$L, coarse rel-tol=$T"
        ORDER+=("$name")
    done
done

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "Phase 3i (LOCALIZED Cg+Schwarz{MG}; 2-D max_levels x coarse rel-tol): $STEPS iters/variant from restart $RESTART"
echo "  LEVELS={$LEVELS}  TOLS={$TOLS}  cap=$COARSE_CAP"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name'"; continue; }
    ml="${spec%% *}"; rest="${spec#* }"; tol="${rest%% |*}"; desc="${spec#*| }"
    run_variant "$name" "$ml" "$tol" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   read: ./peek-grid.py localized-coarse-reltol"
echo "  KEY QUESTION: does localized still beat the global branch on the CURRENT build (§4.11f was cross-build)?"
echo "localized grid done"
