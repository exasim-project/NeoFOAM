#!/bin/bash
#SBATCH --job-name=phase3k-paper-chebyshev-foci
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# PHASE 3k: find lambda_max EMPIRICALLY, then sweep the Chebyshev degree.
#
# WHY THIS EXISTS -- a failed assumption. Phase 3j ran a localized Chebyshev smoother with
# foci = [0.1, 2.0], justifying lambda_max(D^-1 A) <= 2 by Gershgorin. Result: Chebyshev(1) took
# 35.2 iterations where damped Jacobi takes 14.6 -- a 2.4x LOSS where a near-tie was expected
# (degree-1 Chebyshev with those foci is alpha = 1/center ~ 0.95, i.e. damped Jacobi at omega ~ 0.95
# vs the incumbent's 0.9). Those runs are quarantined in discarded-wrong-foci/.
#
# The Gershgorin bound lambda <= 2 requires WEAK DIAGONAL DOMINANCE (an M-matrix). This is a real
# DrivAer mesh: the pressure Laplacian carries NON-ORTHOGONAL correction terms and is NOT guaranteed
# to be an M-matrix, so lambda_max may well exceed 2. UNDERestimating lambda_max is Chebyshev's
# classic failure: the polynomial AMPLIFIES every mode above the assumed bound instead of damping it
# -- degraded-but-not-divergent, exactly the symptom observed. The bound was quoted without checking
# its hypothesis against this operator.
#
# STEP A -- foci sweep AS A MEASUREMENT. Fix degree=1 (where Chebyshev should reduce to damped Jacobi
# and therefore MUST match the ~14.6-iteration baseline if the foci are right) and sweep the upper
# focus. The value at which iterations fall back to baseline IS an estimate of lambda_max: below it
# the polynomial amplifies; above it we merely over-damp (safe, slightly wasteful). So this sweep is
# both the fix and the eigenvalue estimate we should have had.
#
# STEP B -- degree sweep at the lambda_max found in step A (pass FOCI_HI=<value>).
#
# Localized smoother throughout: Schwarz{ Chebyshev + Jacobi } -> all SpMVs hit the rank-local block,
# ZERO halos, so degree is free in communication and costs only (idle) GPU arithmetic. Step 3j
# established this is a wall-time wash at degree 1 (2.376 vs 2.387) -- it buys a 16 %-cheaper V-cycle
# for 17 % more iterations. Everything degree>1 recovers is therefore upside.
#
# Usage:  ./phase3k-paper-study-chebyshev-foci.sh stepA
#         FOCI_HI=8 ./phase3k-paper-study-chebyshev-foci.sh stepB

STUDY_TYPE=chebyshev-foci
STEPS="${STEPS:-50}"
MAXLEVELS="${MAXLEVELS:-3}"          # the no-sc global optimum, so the baseline is known: 13.3 iters
COARSE_TOL="${COARSE_TOL:-0.1}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

BASE_CFG="system/gko/p-multigrid-mgsc-merge2-lcg.json"
CFG="system/gko/cheb-papertmp.json"
trap 'rm -f "$CFG"' EXIT

build() {   # $1 = degree, $2 = foci_lo, $3 = foci_hi
    jq --argjson ml "$MAXLEVELS" --argjson tol "$COARSE_TOL" \
       --argjson d "$1" --argjson lo "$2" --argjson hi "$3" '
        .preconditioner.max_levels = $ml
        | .preconditioner.scale_correction = false
        | .preconditioner.coarsest_solver.local_solver.criteria = [
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":$tol},
            {"type":"Iteration","max_iters":50}
          ]
        | .preconditioner.pre_smoother = [{
            "type":"preconditioner::Schwarz",
            "local_solver":{"type":"solver::Chebyshev","foci":[$lo,$hi],
                            "preconditioner":{"type":"preconditioner::Jacobi","max_block_size":1},
                            "criteria":[{"type":"Iteration","max_iters":$d}]}}]
        | .preconditioner.post_smoother = .preconditioner.pre_smoother
      ' "$BASE_CFG" > "$CFG" || return 1
    jq -e '.preconditioner.pre_smoother[0].local_solver.foci' "$CFG" >/dev/null || return 1
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

# lo = hi/20 keeps the standard smoother band [lambda_max/20, lambda_max] as hi varies.
do_stepA() {
    echo "--- STEP A: foci sweep at DEGREE 1 (must match the damped-Jacobi baseline of ~13.3 iters"
    echo "            at L$MAXLEVELS/tol$COARSE_TOL if the foci are right). Where iters drop back = lambda_max. ---"
    for hi in 1.0 2.0 4.0 8.0 16.0; do
        lo=$(awk "BEGIN{printf \"%.4f\", $hi/20}")
        n=$(echo "$hi" | tr -d '.')
        build 1 "$lo" "$hi" || { echo "   skip hi=$hi"; continue; }
        run_one "fociA-hi${n}" "$TMP_FVSOL" "Chebyshev deg=1, foci [$lo,$hi] -- lambda_max probe"
    done
}

do_stepB() {
    local hi="${FOCI_HI:?set FOCI_HI to the lambda_max found in step A}"
    local lo; lo=$(awk "BEGIN{printf \"%.4f\", $hi/20}")
    echo "--- STEP B: degree sweep at foci [$lo,$hi] ---"
    for d in 1 2 3 4; do
        build "$d" "$lo" "$hi" || { echo "   skip deg=$d"; continue; }
        run_one "chebB-d${d}" "$TMP_FVSOL" "localized Chebyshev degree=$d, foci [$lo,$hi], L$MAXLEVELS/tol$COARSE_TOL"
    done
}

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=(stepA)
for s in "${SEL[@]}"; do
    case "$s" in
        stepA) do_stepA ;;
        stepB) do_stepB ;;
        *) echo "!! unknown step '$s' (stepA stepB)";;
    esac
done

print_summary
echo
echo "baseline to beat: standard Ir smoother, L$MAXLEVELS/tol$COARSE_TOL = 2.377 s/step, 13.3 iters"
echo "chebyshev-foci done"
