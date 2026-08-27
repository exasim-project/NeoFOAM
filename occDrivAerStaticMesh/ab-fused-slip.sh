#!/bin/bash
# A/B: champion config, legacy component-wise U solve vs fused slip solve.
# Reports STEADY per-step wall (median of steps 2..N-1 ExecutionTime deltas), excluding the
# cold-start build (step 1) and the final write step.
STUDY_TYPE=ab-fused-slip
STEPS="${STEPS:-12}"
RESTART="${RESTART:-1050}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

export NEON_MGSC_MODE=post
use_kokkos_tool ""

CHAMP_CFG="system/gko/p-champion-mgscpost-L6-tol01.json"
sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${CHAMP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
    "$MG_TEMPLATE" > "$TMP_FVSOL" || exit 1

steady() {  # $1 = log ; median of interior per-step deltas
    grep "ExecutionTime" "$1" | sed -E 's/.*ExecutionTime = ([0-9.]+) s.*/\1/' \
      | awk 'NR>1{d=$1-p; if(NR>2) print d} {p=$1}' | sort -n \
      | awk '{a[NR]=$1} END{n=NR; if(n%2) printf "%.3f", a[(n+1)/2]; else printf "%.3f",(a[n/2]+a[n/2+1])/2}'
}

for mode in legacy fused; do
    if [ "$mode" = fused ]; then export NEON_FUSED_SLIP_SOLVE=1; else unset NEON_FUSED_SLIP_SOLVE; fi
    export MPIRUN_FORWARD_ENV="NEON_MGSC_MODE NEON_FUSED_SLIP_SOLVE"
    run_one "ab-$mode" "$TMP_FVSOL" "champion U=$mode slip"
    L="${RUN_LOG[ab-$mode]}"
    echo ">>> $mode steady/step = $(steady "$L") s   (cont=$(grep 'sum local' "$L" | tail -1 | sed -E 's/.*sum local = ([0-9.eE+-]+),.*/\1/'))"
done
print_summary
