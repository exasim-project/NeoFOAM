#!/bin/bash
# A/B: champion config with the DSL fused div+laplacian optimizer OFF vs ON (fvSolution
# solvers/<field>/optimize for U,k,omega). Reports STEADY per-step wall (median of interior
# ExecutionTime deltas) + convergence (p iters, continuity) for each mode.
STUDY_TYPE=ab-optimize
STEPS="${STEPS:-12}"
RESTART="${RESTART:-1050}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

export NEON_MGSC_MODE=post
use_kokkos_tool ""

CHAMP_CFG="system/gko/p-champion-mgscpost-L6-tol01.json"
[ -f "$CHAMP_CFG" ] || { echo "!! missing $CHAMP_CFG"; exit 1; }

steady() {  # median of interior per-step ExecutionTime deltas
    grep "ExecutionTime" "$1" | sed -E 's/.*ExecutionTime = ([0-9.]+) s.*/\1/' \
      | awk 'NR>1{d=$1-p; if(NR>2) print d} {p=$1}' | sort -n \
      | awk '{a[NR]=$1} END{n=NR; if(n%2) printf "%.3f", a[(n+1)/2]; else printf "%.3f",(a[n/2]+a[n/2+1])/2}'
}

for mode in false true; do
    # build the champion fvSolution, then set optimize for U/k/omega
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${CHAMP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || exit 1
    for fld in U k omega; do
        foamDictionary -entry "solvers/$fld/optimize" -set "$mode" -disableFunctionEntries "$TMP_FVSOL" >/dev/null 2>&1
    done
    echo ">>> optimize=$mode : $(grep -c "optimize *$mode" "$TMP_FVSOL") fields set"
    run_one "ab-opt-$mode" "$TMP_FVSOL" "champion, optimize=$mode (U/k/omega)"
    L="${RUN_LOG[ab-opt-$mode]}"
    echo "    steady/step = $(steady "$L") s"
    echo "    p iters (last 3): $(grep 'Solving for p,' "$L" | tail -3 | sed -E 's/.*No Iterations ([0-9]+).*/\1/' | tr '\n' ' ')"
    echo "    U init resid Ux (last 2): $(grep 'Solving for Ux,' "$L" | tail -2 | sed -E 's/.*Initial residual = ([0-9.eE+-]+),.*/\1/' | tr '\n' ' ')"
    echo "    continuity (last 3): $(grep 'sum local' "$L" | tail -3 | sed -E 's/.*sum local = ([0-9.eE+-]+),.*/\1/' | tr '\n' ' ')"
done
print_summary
