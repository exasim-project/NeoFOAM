#!/bin/bash
#SBATCH --job-name=investigate-global-scalecorr
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Investigate PERFORMANCE IMPROVEMENTS for GLOBAL multigrid scale correction.
#
# Finding so far (phase3c/phase3b): per-smoother Ir scale_correction (the gm*-sc configs) is a NET
# LOSS on the global Cg+MG path -- it cuts pressure iterations ~15-20% but RAISES per-solve time ~40%
# and its symmetric (fwd+bwd) form can diverge under plain Cg (phase3b cgmg-sc-on stalled at 164
# iters). Root cause (ginkgo core/solver/ir.cpp apply_scale_correction): each corrected smoother sweep
# does an EXTRA SpMV + 2 global dot-product allreduces + an EXTRA full inner-solver apply -> it roughly
# DOUBLES the smoother work at every level.
#
# Cheaper alternative (ginkgo core/solver/multigrid.cpp): the MG-LEVEL `scale_correction: true` boolean
# Rayleigh-scales the smoother output and the prolonged coarse correction at the restriction/
# prolongation boundary (2 SpMV + 4 dots per level) with NO extra smoother apply. It is nonlinear, so
# it needs a FLEXIBLE outer Krylov (FCG), not plain Cg -- which is exactly why per-smoother sc-on
# diverged under Cg. (The prior localized-scalecorr dead-end does NOT apply here: these are GLOBAL, so
# the Rayleigh dots are consistent global allreduces over the whole distributed matrix.)
#
# All CACHED (cacheSolver=true, interval=0), 50 steps from the iter-1000 restart, same session so the
# baselines and candidates see identical GPU/runtime conditions (single-run variance is large -- see
# the gm1 4.56-vs-5.68 swing; treat <~5% gaps as noise).
#
#   name        config                        outer  scale-correction mechanism
#   off         p-multigrid.json              Cg     none                        (baseline)
#   ir-sc       p-multigrid-sc.json           Cg     per-smoother fwd/bwd        (current gm1-sc)
#   fcg-ir-sc   p-fcg-multigrid-sc.json       FCG    per-smoother fwd/bwd        (does FCG rescue it?)
#   mgsc-cg     p-multigrid-mgsc.json         Cg     MG-level (nonlinear)        (diverge check)
#   mgsc-fcg    p-fcg-multigrid-mgsc.json     FCG    MG-level (cheap)            (the candidate)
#
# What to read: p_ms/solve (does MG-level sc avoid the per-sweep doubling?), p_iters (does it still cut
# iterations?), cont + steps (mgsc-cg expected to stall/diverge), s/step (the net verdict vs off=4.5).

STUDY_TYPE=scalecorr-improve
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

run_variant() {
    build_variant_fvsolution "$2" || { echo "   skip $1"; return; }
    run_one "$1" "$TMP_FVSOL" "$3"
}

declare -A VARIANT=(
    [off]="p-multigrid.json | GLOBAL Cg+MG, CACHED, NO scale correction (baseline)"
    [ir-sc]="p-multigrid-sc.json | GLOBAL Cg+MG, CACHED, per-smoother scale correction fwd/bwd (current)"
    [fcg-ir-sc]="p-fcg-multigrid-sc.json | GLOBAL FCG+MG, CACHED, per-smoother scale correction fwd/bwd (FCG-rescue)"
    [mgsc-cg]="p-multigrid-mgsc.json | GLOBAL Cg+MG, CACHED, MG-LEVEL scale_correction=true (needs FCG -- diverge check)"
    [mgsc-fcg]="p-fcg-multigrid-mgsc.json | GLOBAL FCG+MG, CACHED, MG-LEVEL scale_correction=true (candidate)"
)
ORDER=(off ir-sc fcg-ir-sc mgsc-cg mgsc-fcg)

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "scalecorr-improve: $STEPS iterations/variant from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    cfg="${spec%% *}"; desc="${spec#*| }"
    run_variant "$name" "$cfg" "$desc"
done

print_summary
echo
echo "logs under: $RESULTS/   |   goal: an sc form that cuts p_iters WITHOUT the per-sweep time blow-up"
