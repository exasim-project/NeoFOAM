#!/bin/bash
#
# Cost break-down of the CURRENT CHAMPION (corrected-binding data, 2026-07-17):
#   Global MG + scale-correction POST-pass, max_levels=6, coarse rel-tol 0.1, pgmMerge2, CACHED.
#   Best pressure-solve cell of the corrected rel-tol grids: 151 ms p-solve, 8.4 outer iters,
#   0.790 steady s/step. Needs NEON_MGSC_MODE=post (post-only scale-correction pass).
#
# Same instruments as phase1 (which profiles the REFERENCE), but on the champion config and in its
# ACTUAL operating mode (cacheSolver=true, interval=0) rather than the reference's rebuild-every-solve:
#   spacetimestack  region wall-time tree + GPU-vs-host split  -> where the champion step spends time
#   memhwm          device high-water-mark                     -> champion peak
#   memtimeline     NF_MEM_SCOPE per-region device-pool CSV     -> ranked by plot_memory_timeline.py
#
# Runs from the 1000-restart, RESTART -> RESTART+STEPS window. Requires the Phase-0 restart.
#
# Usage:  ./champion-costbreakdown.sh                 # all three instruments
#         ./champion-costbreakdown.sh spacetimestack  # one

STUDY_TYPE=champion-costbreakdown
STEPS="${STEPS:-30}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

# scale-correction POST-only pass on all ranks (the champion's mode)
export NEON_MGSC_MODE=post

BASE_CFG="system/gko/p-multigrid-mgsc-merge2-lcg.json"
CHAMP_CFG="system/gko/p-champion-mgscpost-L6-tol01.json"
CELLS_TOTAL="${CELLS_TOTAL:-65334765}"
CELLS_PER_RANK=$(( CELLS_TOTAL / NP ))
MEMTOOLS="$(dirname "$0")/../examples/neoSimpleFoam"   # plot_memory_timeline.py

# Build the champion config = phase3h SC=post L6 tol0.1 (max_levels=6, sc=true, pgmMerge2,
# coarse ResidualNorm(0.1)+Iteration(50)). Written once, persistent.
build_champion_config() {
    [ -f "$BASE_CFG" ] || { echo "!! missing $BASE_CFG" >&2; return 1; }
    jq '.preconditioner.max_levels = 6
        | .preconditioner.scale_correction = true
        | .preconditioner.mg_level = ["neon::pgmMerge2"]
        | .preconditioner.coarsest_solver.local_solver.criteria = [
            {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":0.1},
            {"type":"Iteration","max_iters":50}
          ]' "$BASE_CFG" > "$CHAMP_CFG" || return 1
    echo "   champion config -> $CHAMP_CFG (L6, tol0.1, sc-post, merge2)"
}

build_champion_fvsolution() {
    # cacheSolver=true + interval=0 : the champion's real operating mode (build hierarchy once, reuse).
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${CHAMP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

collect_mem_output() {
    local name="$1" log="${RUN_LOG[$name]}" stem
    [ -n "$log" ] || return 0; stem="${log%.log}"
    [ -f memoryTimeline.csv ] || { echo "   (no memoryTimeline.csv -- skipped or probing off)"; return 0; }
    mv memoryTimeline.csv "${stem}.memoryTimeline.csv"
    if python3 "$MEMTOOLS/plot_memory_timeline.py" "${stem}.memoryTimeline.csv" \
            -o "${stem}.memoryTimeline.png" -n "$CELLS_PER_RANK" > "${stem}.memoryTimeline.txt" 2>&1; then
        echo "   mem timeline -> ${stem}.memoryTimeline.{csv,png,txt}"
    else
        echo "   !! plot_memory_timeline.py failed (see ${stem}.memoryTimeline.txt)"
    fi
}

build_champion_config      || { echo "!! could not build champion config"; exit 1; }
build_champion_fvsolution  || { echo "!! could not build champion fvSolution"; exit 1; }

INSTR=("$@"); [ ${#INSTR[@]} -eq 0 ] && INSTR=(spacetimestack memhwm memtimeline)
echo "champion cost break-down: sc-post L6/tol0.1 (cached), $STEPS-step window from restart $RESTART"
echo "  instruments: ${INSTR[*]}"

for i in "${INSTR[@]}"; do
    case "$i" in
        spacetimestack|stack)
            use_kokkos_tool space-time-stack
            export MPIRUN_FORWARD_ENV="NEON_MGSC_MODE"
            run_one "champ-spacetimestack" "$TMP_FVSOL" "champion sc-post L6/tol0.1, region wall-time break-down"
            use_kokkos_tool ""; unset MPIRUN_FORWARD_ENV ;;
        memhwm|hwm)
            use_kokkos_tool memory-high-water-mark
            export MPIRUN_FORWARD_ENV="NEON_MGSC_MODE"
            run_one "champ-memhwm" "$TMP_FVSOL" "champion sc-post L6/tol0.1, device high-water-mark"
            use_kokkos_tool ""; unset MPIRUN_FORWARD_ENV ;;
        memtimeline|timeline)
            use_kokkos_tool ""
            export NEOFOAM_MEM_TIMELINE=1 NEOFOAM_MEM_TIMELINE_FILE=memoryTimeline.csv
            export MPIRUN_FORWARD_ENV="NEON_MGSC_MODE NEOFOAM_MEM_TIMELINE NEOFOAM_MEM_TIMELINE_FILE"
            run_one "champ-memtimeline" "$TMP_FVSOL" "champion sc-post L6/tol0.1, NF_MEM_SCOPE device-pool timeline"
            collect_mem_output "champ-memtimeline"
            unset NEOFOAM_MEM_TIMELINE NEOFOAM_MEM_TIMELINE_FILE MPIRUN_FORWARD_ENV ;;
        *) echo "!! unknown instrument '$i'" ;;
    esac
done

print_summary
echo
echo "champion cost-break-down artifacts under: $RESULTS/"
echo "  region wall-time -> champ-spacetimestack-*.kokkos-profile.txt"
echo "  memory HWM       -> champ-memhwm-*.log"
echo "  memory timeline  -> champ-memtimeline-*.memoryTimeline.{csv,png,txt}"
