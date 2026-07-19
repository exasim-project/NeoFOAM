#!/bin/bash
#SBATCH --job-name=comm-attribution
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# WHERE DOES THE COMMUNICATION COME FROM: the smoother or the scale correction?
#
# The two mechanisms emit DIFFERENT collective types, so they are separable:
#   * every distributed SpMV -> RowGatherer halo exchange -> MPI_Alltoallv
#       sources: smoother's residual (Ir: r=b-Ax), restriction, prolongation, coarse solve, AND
#                scale correction's A*delta
#   * every dot / norm -> MPI_Allreduce
#       sources: outer Cg dots, stopping criteria, coarse Cg dots, AND scale correction's Rayleigh
#                dots  sf = (d.b)/(d.Ad)   [multigrid.cpp:678, :760]
#
# scale_correction is a pure config flag, so tracing sc=ON vs sc=OFF at otherwise IDENTICAL settings
# isolates its contribution exactly; the sc=OFF run is the smoother+transfers+coarse baseline.
#
# NOTE ON NORMALIZATION: sc changes the OUTER iteration count (§4.11b: 16.5 -> 6), so RAW collective
# totals are NOT comparable. Everything must be normalized PER OUTER CG ITERATION (= per V-cycle),
# which the analysis script does by dividing by the summed "No Iterations" from the log.
#
# Usage:  ./investigate-comm-attribution.sh          # both variants
#         ./investigate-comm-attribution.sh scon     # single
# Then:   python3 analyze-comm-attribution.py

STUDY_TYPE=comm-attribution
STEPS="${STEPS:-10}"        # short window; we count collectives, not wall time
MAXLEVELS="${MAXLEVELS:-4}" # the §4.11k best depth
COARSE_TOL="${COARSE_TOL:-0.1}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

BASE_CFG="system/gko/p-multigrid-mgsc-merge2-lcg.json"
SWEEP_CFG="system/gko/commattr-papertmp.json"
cleanup_sweep_cfg() { rm -f "$SWEEP_CFG"; }
trap cleanup_sweep_cfg EXIT

build_variant_config() {
    # $1 = scale_correction (true|false). Pin max_levels + the rel-tol coarse criterion; toggle ONLY sc.
    local sc="$1"
    jq --argjson ml "$MAXLEVELS" --argjson tol "$COARSE_TOL" --argjson sc "$sc" '
        .preconditioner.max_levels = $ml
        | .preconditioner.scale_correction = $sc
        | .preconditioner.coarsest_solver.local_solver.criteria = [
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":$tol},
            {"type":"Iteration","max_iters":50}
          ]' "$BASE_CFG" > "$SWEEP_CFG" || return 1
    local got; got=$(jq '.preconditioner.scale_correction' "$SWEEP_CFG")
    [ "$got" = "$sc" ] || { echo "!! failed to set scale_correction=$sc (got $got)" >&2; return 1; }
}

build_variant_fvsolution() {
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${SWEEP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

run_variant() {
    local name="$1" sc="$2" desc="$3"
    command -v nsys >/dev/null 2>&1 || { echo "!! nsys not on PATH (module load cuda) -- abort"; return 1; }
    build_variant_config "$sc"  || { echo "   skip $name"; return; }
    build_variant_fvsolution    || { echo "   skip $name"; return; }
    local stem="$RESULTS/${name}-nsysmpi-$(date +%Y%m%d-%H%M%S)"
    # --trace=mpi is what populates MPI_COLLECTIVES_EVENTS; cuda for the stream context.
    export LAUNCH_WRAPPER="nsys profile --trace=cuda,nvtx,mpi --sample=none --cpuctxsw=none -f true -o ${stem}.rank%q{OMPI_COMM_WORLD_RANK}"
    run_one "$name" "$TMP_FVSOL" "$desc"
    unset LAUNCH_WRAPPER
    echo "   trace: ${stem}.rank0.nsys-rep"
}

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=(scon scoff)
echo "Comm attribution (max_levels=$MAXLEVELS, coarse rel-tol=$COARSE_TOL): $STEPS steps from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    case "$name" in
        scon)  run_variant scon  true  "MG-level scale_correction ON  (smoother+transfers+coarse+SC)" ;;
        scoff) run_variant scoff false "MG-level scale_correction OFF (smoother+transfers+coarse only)" ;;
        *) echo "!! unknown variant '$name' (scon scoff)";;
    esac
done

print_summary
echo
echo "traces under: $RESULTS/   |   analyze: python3 analyze-comm-attribution.py"
echo "comm-attribution done"
