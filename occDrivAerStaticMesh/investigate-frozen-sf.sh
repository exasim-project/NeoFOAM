#!/bin/bash
#SBATCH --job-name=investigate-frozen-sf
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Prototype study: FROZEN Rayleigh scale-correction factors in Ginkgo MG (multigrid.cpp).
#
# MG-level scale_correction=true (the §4.11 win) recomputes the Rayleigh factor sf=(δ·b)/(δ·Aδ) on
# EVERY V-cycle at EVERY level, via 2 compute_dot allreduces per correction point (down + up). On this
# latency/dispatch-bound case that is ~200 tiny allreduces per pressure solve purely to (re)compute sf.
# The prototype patch (build/.../ginkgo-src/core/solver/multigrid.cpp, env-gated) lets sf be computed
# once and reused:
#
#   GKO_SC_RECOMPUTE_INTERVAL = 1   recompute every V-cycle          (DEFAULT = original behaviour)
#                             = 0   compute once, freeze until the solver is regenerated
#                             = N   recompute every N-th V-cycle, reuse in between
#
# The frozen sf and its per-level "was the correction active" flag live in MultigridState (persists in
# cache_.state across V-cycles and across SIMPLE steps until update_matrix_value regenerates). The SpMV
# (A·δ) and the extra smoother apply STILL run when frozen — only the 2 dot-allreduces per point are
# skipped — so this trims allreduce LATENCY, not the correction's compute.
#
# All variants use the SAME config (p-multigrid-mgsc.json = Cg + global MG + MG-level sc + plain Pgm,
# the §4.11 winner), CACHED, 50 steps from the iter-1000 restart, same session. Only the env var differs.
#
#   sf1  interval=1  recompute every cycle   (regression baseline: MUST reproduce mgsc-cg ~6 iters)
#   sf0  interval=0  freeze after first       (max saving, max risk)
#   sf2  interval=2  refresh every 2nd cycle
#   sf4  interval=4  refresh every 4th cycle
#   sf8  interval=8  refresh every 8th cycle
#
# What to read:
#   p_iters   THE risk metric — does a stale sf inflate the outer iteration count? (sf1 ~6 is the ref)
#   p_ms      the reward — fewer allreduces should cut per-solve time IF iterations hold
#   s/step    net verdict; cont must stay ~3e-6 (a frozen sf that diverges shows up here + p_iters->1000)
#
# NOTE the 3-step smoke already showed interval=0 hitting the 1000-iter cap (cont 1.4e-4): freezing sf
# from cycle 1 of a solve is likely too aggressive because early-iteration sf differs from the settled
# value. The interval>=2 variants (refresh periodically) are the interesting middle ground.

STUDY_TYPE=frozen-sf
STEPS="${STEPS:-50}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"

require_restart

# Forward the interval env var to every MPI rank (kokkos_launch -x's names listed here).
export MPIRUN_FORWARD_ENV="GKO_SC_RECOMPUTE_INTERVAL"

CFG="p-multigrid-mgsc.json"

build_variant_fvsolution() {
    local cfg="$1" cache="true" interval="0"
    if [ ! -f "system/gko/$cfg" ]; then echo "!! missing system/gko/$cfg" >&2; return 1; fi
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$cfg;\n        cacheSolver      $cache;\n        preconditionerRebuildInterval $interval;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

# $1 = variant name, $2 = GKO_SC_RECOMPUTE_INTERVAL value, $3 = desc
run_sf() {
    build_variant_fvsolution "$CFG" || { echo "   skip $1"; return; }
    export GKO_SC_RECOMPUTE_INTERVAL="$2"
    echo "   GKO_SC_RECOMPUTE_INTERVAL=$GKO_SC_RECOMPUTE_INTERVAL"
    run_one "$1" "$TMP_FVSOL" "$3"
}

declare -A INTERVAL=( [sf1]=1 [sf0]=0 [sf2]=2 [sf4]=4 [sf8]=8 )
declare -A DESC=(
    [sf1]="MG-level sc, recompute sf EVERY cycle (regression baseline == mgsc-cg)"
    [sf0]="MG-level sc, FREEZE sf after first cycle"
    [sf2]="MG-level sc, refresh sf every 2nd cycle"
    [sf4]="MG-level sc, refresh sf every 4th cycle"
    [sf8]="MG-level sc, refresh sf every 8th cycle"
)
ORDER=(sf1 sf0 sf2 sf4 sf8)

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "frozen-sf: $STEPS iterations/variant from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    iv="${INTERVAL[$name]:-}"
    [ -n "$iv" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    run_sf "$name" "$iv" "${DESC[$name]}"
done
unset GKO_SC_RECOMPUTE_INTERVAL

print_summary
echo
echo "logs under: $RESULTS/   |   sf1 (=6 iters) is the regression ref; does interval>=2 cut p_ms without inflating p_iters?"
