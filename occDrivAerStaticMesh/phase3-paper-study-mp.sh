#!/bin/bash
#
# Phase 3c — MIXED-PRECISION sweep for the multigrid PRECONDITIONER (the [pending] Reproduce entry).
#
# Baseline is the CURRENT CHAMPION (corrected-binding data, §4.15): global Cg + Multigrid with the
# MG-level scale-correction POST pass, max_levels=6, coarse rel-tol 0.1, pgmMerge2, CACHED
# (cacheSolver=true, preconditionerRebuildInterval=0), NEON_MGSC_MODE=post. All arms are that exact
# config; they differ ONLY in a Ginkgo per-node `value_type` override, so any delta is precision, not
# structure. Every arm runs the identical 1000 -> 1000+STEPS restart window (apples-to-apples).
#
#   arm            value_type override on the champion config          precision boundary
#   ------------   -------------------------------------------------   ----------------------------
#   fp64           (none)                                              baseline / denominator
#   mgFloat32      .preconditioner.value_type = "float32"             whole V-cycle in fp32
#   coarseFloat32  .preconditioner.coarsest_solver.value_type=float32 only the coarsest solve in fp32
#   coarseBf16     .preconditioner.coarsest_solver.value_type=bf16    only the coarsest solve in bf16
#
# Optional de-localized-coarse fallback arms (COARSE_DELOC=1): the champion's coarsest solver is a
# distributed preconditioner::Schwarz{Cg+Jacobi}. Prior evidence ([[neon-mixedprec-distributed-schwarz]])
# is that reduced precision on a distributed Schwarz can abort at setup (bf16 absent from the Schwarz
# value_type_list; float gko::as NotSupported on the fp64 distributed matrix). If coarseFloat32/coarseBf16
# crash at setup, COARSE_DELOC=1 also runs two arms whose coarse solver is a PLAIN distributed Cg+Jacobi
# (Schwarz dropped) so the precision cast avoids the Schwarz value_type_list limitation.
#
# Usage:
#   ./phase3-paper-study-mp.sh                       # fp64 + 3 precision arms, 30-step window
#   STEPS=50 ./phase3-paper-study-mp.sh              # longer window
#   COARSE_DELOC=1 ./phase3-paper-study-mp.sh        # + de-localized-coarse fallback arms
#   FORCE=1 ./phase3-paper-study-mp.sh               # re-run even if logs exist
#   ARMS="fp64 mgFloat32" ./phase3-paper-study-mp.sh # run a subset

STUDY_TYPE=mixed-precision
STEPS="${STEPS:-30}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

# The champion runs the MG scale-correction as a POST-only pass on all ranks; forward it to every rank.
export NEON_MGSC_MODE=post
export MPIRUN_FORWARD_ENV="NEON_MGSC_MODE"

GKO=system/gko
BASE_CFG="$GKO/p-multigrid-mgsc-merge2-lcg.json"   # sc + pgmMerge2 + localized-coarse-cg base
CHAMP_CFG="$GKO/p-champion-mgscpost-L6-tol01.json" # fp64 champion (built from BASE_CFG below)

# ---- build the fp64 champion config from the base (same recipe as champion-costbreakdown.sh) ----
build_champion_config() {
    [ -f "$BASE_CFG" ] || { echo "!! missing $BASE_CFG" >&2; return 1; }
    jq '.preconditioner.max_levels = 6
        | .preconditioner.scale_correction = true
        | .preconditioner.mg_level = ["neon::pgmMerge2"]
        | .preconditioner.coarsest_solver.local_solver.criteria = [
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":0.1},
            {"type":"Iteration","max_iters":50}
          ]' "$BASE_CFG" > "$CHAMP_CFG" || return 1
}

# ---- derive a precision variant from the champion via a jq filter; echoes the config path ----
# $1 = arm name, $2 = jq filter ('.' = identity = fp64 baseline)
make_variant_cfg() {
    # NB: plain (non-dot) basename -- OpenFOAM's dict parser rejects a configFile path VALUE with a
    # leading-dot component ("ill defined primitiveEntry"), so the generated configs must not be dotfiles.
    local arm="$1" filter="$2" out="$GKO/mp-${arm}.json"
    jq "$filter" "$CHAMP_CFG" > "$out" || { echo "!! jq failed for $arm" >&2; return 1; }
    echo "$out"
}

# ---- champion fvSolution (cacheSolver=true, interval=0) pointing at a given p configFile ----
# $1 = config path, $2 = output fvSolution path
build_fvsolution_for() {
    local cfg="$1" out="$2"
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${cfg};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$out"
}

build_champion_config || { echo "!! could not build champion config"; exit 1; }

# ---- arm registry: name | jq filter (config edit) | description -----------------------------------
declare -A ARM_FILTER ARM_DESC
ARM_ORDER=(fp64 mgFloat32 coarseFloat32 coarseBf16)

ARM_FILTER[fp64]='.'
ARM_DESC[fp64]='baseline champion (fp64 MG), denominator'

ARM_FILTER[mgFloat32]='.preconditioner.value_type = "float32"'
ARM_DESC[mgFloat32]='whole MG preconditioner (V-cycle) in float32, outer Cg stays fp64'

ARM_FILTER[coarseFloat32]='.preconditioner.coarsest_solver.value_type = "float32"'
ARM_DESC[coarseFloat32]='only the coarsest solve (Schwarz{Cg+Jacobi}) in float32'

ARM_FILTER[coarseBf16]='.preconditioner.coarsest_solver.value_type = "bfloat16"'
ARM_DESC[coarseBf16]='only the coarsest solve (Schwarz{Cg+Jacobi}) in bfloat16'

# Optional de-localized-coarse fallback arms: replace the Schwarz coarse solver with a plain
# distributed Cg+Jacobi (same criteria), THEN drop the reduced precision on that non-Schwarz node.
if [ -n "${COARSE_DELOC:-}" ]; then
    _deloc='.preconditioner.coarsest_solver = {
        "type":"solver::Cg",
        "preconditioner":{"type":"preconditioner::Jacobi","max_block_size":1},
        "criteria":[
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":0.1},
            {"type":"Iteration","max_iters":50}
        ]}'
    ARM_FILTER[coarseFloat32Deloc]="$_deloc | .preconditioner.coarsest_solver.value_type = \"float32\""
    ARM_DESC[coarseFloat32Deloc]='de-localized coarse (plain distributed Cg+Jacobi) in float32'
    ARM_FILTER[coarseBf16Deloc]="$_deloc | .preconditioner.coarsest_solver.value_type = \"bfloat16\""
    ARM_DESC[coarseBf16Deloc]='de-localized coarse (plain distributed Cg+Jacobi) in bfloat16'
    ARM_ORDER+=(coarseFloat32Deloc coarseBf16Deloc)
fi

# Allow an explicit subset via ARMS="..."
[ -n "${ARMS:-}" ] && read -r -a ARM_ORDER <<< "$ARMS"

echo "mixed-precision sweep (champion sc-post L6/tol0.1/merge2, cached): $STEPS-step window from restart $RESTART"
echo "  build: $NEON_BUILD$NEON_DEVICE   arms: ${ARM_ORDER[*]}"
echo

for arm in "${ARM_ORDER[@]}"; do
    filter="${ARM_FILTER[$arm]}"
    [ -n "$filter" ] || { echo "!! unknown arm '$arm' -- skipping"; continue; }
    cfg=$(make_variant_cfg "$arm" "$filter") || continue
    fvsol="$CFGDIR/.fvSolution.mp-${arm}"
    build_fvsolution_for "$cfg" "$fvsol"
    run_one "$arm" "$fvsol" "${ARM_DESC[$arm]}"
done

print_summary

cat <<'EON'

Notes:
  * s/step in the summary is setup-INCLUSIVE (ExecutionTime/steps), the same-window denominator used
    throughout the paper study (§4.6). For the marginal steady per-step, use (ET_last-ET_first)/(steps-1).
  * An arm that aborts at setup (exit != 0, steps < requested) is a RESULT, not a harness failure --
    it confirms the distributed-Schwarz reduced-precision block for that node ([[neon-mixedprec-distributed-schwarz]]).
    Re-run with COARSE_DELOC=1 to test the de-localized-coarse fallback.
  * Correctness lives in the `cont` (continuity, sum local) column and the pressure iteration count
    (p_iters/it): a precision arm that converges to the SAME iters/continuity as fp64 is "free".
EON
