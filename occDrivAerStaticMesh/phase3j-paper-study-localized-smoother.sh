#!/bin/bash
#SBATCH --job-name=phase3j-paper-localized-smoother
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# PHASE 3j: localize the SMOOTHER, then make it a Chebyshev polynomial.
#
# ---------------------------------------------------------------------------------------------
# STEP 1 (gate)  -- is smoothing local?
#
# The current smoother is  Ir(0.9) { Schwarz{ Jacobi(1) } }  with Ir OUTSIDE the Schwarz. Point
# Jacobi is local and emits NO collectives; the smoother's entire communication is **Ir's residual**
# r = b - A*x, computed with the GLOBAL distributed A -> one halo per application, x2 (pre+post) per
# level. Against a measured ~19.9 Alltoallv/V-cycle those residuals are plausibly ~30-40 % of it.
#
# Moving Ir INSIDE the Schwarz -> Schwarz{ Ir(0.9){Jacobi(1)} } makes the residual use each rank's
# LOCAL diagonal block A_ii: zero halos in the smoother.
#
# Hypothesis: this is nearly free numerically, because SMOOTHING IS INHERENTLY LOCAL -- the smoother
# targets high-frequency (short-wavelength) error, which does not couple across a 49 M-cell subdomain.
# Only ~1 % of fine-level unknowns sit on a rank interface (surface-to-volume; ~40 % at coarse levels).
# Precedent: localizing the COARSE solve (§4.11i) is theoretically *more* dubious (long-wavelength
# error does span the domain) and still cost ZERO iterations at 4 ranks.
#
# READ: if p-iters is ~unchanged vs the baseline, the hypothesis holds and we banked a comm cut.
#       If p-iters rises materially, smoothing is NOT local here and STEP 2 is not worth running.
#
# ---------------------------------------------------------------------------------------------
# STEP 2 -- localized CHEBYSHEV smoother (gated on step 1)
#
# With the smoother localized, a Chebyshev polynomial of degree m costs m LOCAL SpMVs and ZERO halos
# -- degree becomes free in communication. Since the GPU is ~80 % idle (§4.3), that converts the
# classic comm-for-iterations trade (which scale correction LOST, §5.3) into an
# ARITHMETIC-for-iterations trade -- spending only the resource in surplus.
#
# Chebyshev is also REDUCTION-FREE (verified: zero compute_dot/compute_norm in chebyshev.cpp; the
# alpha/beta coefficients come from the static `foci` on the host). Unlike scale correction, which
# pays 20.1 Allreduce/V-cycle -- global barriers that cannot overlap -- it adds no synchronization.
#
# foci = {lower, upper} eigenvalue bounds of the PRECONDITIONED system D^-1 A. For a smoother one
# targets the top of the spectrum: [lambda_max/ratio, lambda_max], ratio ~20 (cf. MueLu's default).
# lambda_max <= 2 by Gershgorin for a diagonally-scaled weakly-diagonally-dominant M-matrix (the
# pressure Laplacian), so foci = [0.1, 2.0] is safe. Safety matters: UNDERestimating lambda_max makes
# the polynomial AMPLIFY the modes above it -> divergence (not a graceful failure). Bounds stay valid
# per-rank: A_ii is a principal submatrix, so lambda_max(A_ii) <= lambda_max(A) (interlacing).
#
# EXPECT A DEGREE CEILING, not monotone gains: a degree-m polynomial in A_ii propagates information
# ~m cells, so with ZERO Schwarz overlap the missing interface coupling matters more as m grows
# (the classic fix is overlap >= m). Finding where it stops helping IS the result.
#
# Usage:  ./phase3j-paper-study-localized-smoother.sh step1
#         ./phase3j-paper-study-localized-smoother.sh step2
#         ./phase3j-paper-study-localized-smoother.sh            # both

STUDY_TYPE=localized-smoother
STEPS="${STEPS:-50}"
FOCI_LO="${FOCI_LO:-0.1}"       # lambda_max / 20
FOCI_HI="${FOCI_HI:-2.0}"       # Gershgorin bound for D^-1 A
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

BASE_CFG="system/gko/p-multigrid-mgsc-merge2-lcg.json"
CFG="system/gko/locsmooth-papertmp.json"
trap 'rm -f "$CFG"' EXIT

# Every variant sits at the §5.1 champion cell: sc OFF, coarse rel-tol 0.1, localized-CG coarse.
common_jq() {   # $1 = max_levels
    jq --argjson ml "$1" '
        .preconditioner.max_levels = $ml
        | .preconditioner.scale_correction = false
        | .preconditioner.coarsest_solver.local_solver.criteria = [
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":0.1},
            {"type":"Iteration","max_iters":50}
          ]' "$BASE_CFG"
}

# --- smoother blocks -------------------------------------------------------------------------
# baseline: Ir OUTSIDE Schwarz -> residual on the GLOBAL A (one halo per application)
smoother_global() {
    echo '{"type":"solver::Ir","relaxation_factor":0.9,
           "solver":{"type":"preconditioner::Schwarz",
                     "local_solver":{"type":"preconditioner::Jacobi","max_block_size":1}},
           "criteria":[{"type":"Iteration","max_iters":1}]}'
}
# step 1: Ir INSIDE Schwarz -> residual on the LOCAL block A_ii (no halo)
smoother_local() {
    echo '{"type":"preconditioner::Schwarz",
           "local_solver":{"type":"solver::Ir","relaxation_factor":0.9,
                           "solver":{"type":"preconditioner::Jacobi","max_block_size":1},
                           "criteria":[{"type":"Iteration","max_iters":1}]}}'
}
# step 2: localized Chebyshev of degree $1 (degree = the Iteration count)
smoother_cheb() {
    jq -nc --argjson d "$1" --argjson lo "$FOCI_LO" --argjson hi "$FOCI_HI" '
      {"type":"preconditioner::Schwarz",
       "local_solver":{"type":"solver::Chebyshev",
                       "foci":[$lo,$hi],
                       "preconditioner":{"type":"preconditioner::Jacobi","max_block_size":1},
                       "criteria":[{"type":"Iteration","max_iters":$d}]}}'
}

build_cfg() {   # $1 = max_levels, $2 = smoother json
    common_jq "$1" | jq --argjson sm "$2" '.preconditioner.pre_smoother = [$sm]
                                          | .preconditioner.post_smoother = [$sm]' > "$CFG" || return 1
    jq -e '.preconditioner.pre_smoother[0].type' "$CFG" >/dev/null || return 1
}

build_fvsol() {
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

run_variant() {  # $1 name, $2 max_levels, $3 smoother json, $4 desc
    build_cfg "$2" "$3" || { echo "   skip $1 (config build failed)"; return; }
    build_fvsol         || { echo "   skip $1"; return; }
    run_one "$1" "$TMP_FVSOL" "$4"
}

do_step1() {
    echo "--- STEP 1: is smoothing local? (L3, tol 0.1, sc off) ---"
    run_variant "sm-global" 3 "$(smoother_global)" "BASELINE: Ir outside Schwarz (residual on global A, 1 halo/application)"
    run_variant "sm-local"  3 "$(smoother_local)"  "LOCALIZED: Ir inside Schwarz (residual on local A_ii, no halo)"
    echo "--- read: did p-iters move? unchanged => smoothing is local, step 2 is worth running ---"
}

do_step2() {
    echo "--- STEP 2: localized Chebyshev, degree x max_levels (tol 0.1, sc off, foci [$FOCI_LO,$FOCI_HI]) ---"
    for L in 2 3 4; do
        for d in 1 2 3 4; do
            run_variant "cheb${d}-L${L}" "$L" "$(smoother_cheb "$d")" \
                "localized Chebyshev degree=$d, max_levels=$L, foci [$FOCI_LO,$FOCI_HI]"
        done
    done
}

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=(step1 step2)
for s in "${SEL[@]}"; do
    case "$s" in
        step1) do_step1 ;;
        step2) do_step2 ;;
        *) echo "!! unknown step '$s' (step1 step2)";;
    esac
done

print_summary
echo
echo "logs: $RESULTS/   |   reference: no-sc L3/tol0.1 = 2.377 s/step, 13.3 iters (steady state)"
echo "localized-smoother done"
