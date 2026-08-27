#!/bin/bash
# Clean (UNPROFILED) champion s/step, comparable to the 0.79 baseline.
# No kokkos tool attached (space-time-stack adds ~3x host overhead), 30-step window so the
# cold-start MG build amortizes. Champion config = sc-post L6 tol0.1 merge2, cached.
STUDY_TYPE=champion-clean
STEPS="${STEPS:-30}"
RESTART="${RESTART:-1050}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

export NEON_MGSC_MODE=post
export MPIRUN_FORWARD_ENV="NEON_MGSC_MODE"
use_kokkos_tool ""   # NO profiler

CHAMP_CFG="system/gko/p-champion-mgscpost-L6-tol01.json"
[ -f "$CHAMP_CFG" ] || { echo "!! missing $CHAMP_CFG (run champion-costbreakdown.sh once)"; exit 1; }

# cacheSolver=true interval=0 (champion operating mode)
sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${CHAMP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
    "$MG_TEMPLATE" > "$TMP_FVSOL" || exit 1

run_one "champ-clean" "$TMP_FVSOL" "champion sc-post L6/tol0.1, CLEAN (no profiler), ${STEPS}-step"
print_summary
