#!/bin/bash
#SBATCH --job-name=phase1-paper-costbreakdown
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase1-paper-study-costbreakdown.sh
#              NSYS=1 sbatch --export=ALL,NSYS=1 phase1-paper-study-costbreakdown.sh   # add nsys trace
# Or run live: salloc -w gpu-nvidia-h200-3 -n32 -t 08:00:00 --gres gpu:4   then   ./phase1-paper-study-costbreakdown.sh
# (the #SBATCH lines are inert comments when run interactively; override any at submit. Needs the
#  Phase-0 restart -- require_restart aborts loudly if it is missing.)
#
# Optimization-paper study -- PHASE 1: computational cost break-down & detailed profiling of the
# REFERENCE (fp64 PCG + global Multigrid, system/gko/p-multigrid.json).
#
# Runs the reference config from the semi-converged restart (RESTART -> RESTART+STEPS) under each
# instrument below, so we can decompose WHERE the reference spends its time before deciding what to
# optimize. This phase is ALLOWED to carry profiler overhead -- its outputs are RELATIVE attributions,
# not the headline wall time. The overhead-free time-per-timestep is collected separately, AFTER this
# phase, by phase2-paper-study-reference.sh (Phase 2).
#
# Instruments (each a separate run, same config + same restart window):
#   spacetimestack  KOKKOS_TOOL=space-time-stack -> per-region wall-time tree (momentumPredictor,
#                   pressure assemble vs. solve, turbulence, halo exchange, I/O) + GPU-vs-host split.
#                   The primary "where does the time go" artifact. -> <stem>.kokkos-profile.txt
#   memhwm          KOKKOS_TOOL=memory-high-water-mark -> device high-water-mark, the reference peak.
#   memtimeline     NEOFOAM_MEM_TIMELINE=1 -> the NF_MEM_SCOPE per-region device-pool CSV timeline,
#                   ranked by plot_memory_timeline.py. -> <stem>.memoryTimeline.{csv,png,txt}
#   nsys            (opt-in, NSYS=1) LAUNCH_WRAPPER=nsys profile -> a kernel-level trace of the whole
#                   window for drilling into the hottest region. -> <stem>.nsys-rep. Off by default.
#
# Per-solve pressure/U/k/omega iteration counts and continuity error come for free in every log and
# are tabulated by print_summary (the "solver-cost isolation" numbers).
#
# Usage:   ./phase1-paper-study-costbreakdown.sh                 # spacetimestack, memhwm, memtimeline
#          ./phase1-paper-study-costbreakdown.sh spacetimestack  # a single instrument
#          NSYS=1 ./phase1-paper-study-costbreakdown.sh nsys     # add the (heavy) nsys trace
#          FORCE=1 ./phase1-paper-study-costbreakdown.sh         # re-run instead of skipping existing
#
# Requires the restart from Phase 0 (phase0-paper-study-spinup.sh); aborts loudly if it is missing.

STUDY_TYPE=cost-breakdown
# sbatch copies the script into a spool dir, so $0/BASH_SOURCE point there, not at the
# real script. Fall back to $PWD (guaranteed correct by #SBATCH --chdir) when the sibling
# common script isn't next to us.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"

require_restart

REF_FVSOL="$(build_reference_fvsolution)" || { echo "!! could not build reference fvSolution"; exit 1; }

# Mesh size for per-cell memory normalisation in the timeline plot (constant/polyMesh/owner note).
CELLS_TOTAL="${CELLS_TOTAL:-65334765}"
CELLS_PER_RANK=$(( CELLS_TOTAL / NP ))
MEMTOOLS="$(dirname "$0")/../examples/neoSimpleFoam"   # plot_memory_timeline.py

# ---------------------------------------------------------------- instruments
run_spacetimestack() {
    # Per-region wall-time tree + GPU-vs-host split. The primary cost-break-down artifact.
    use_kokkos_tool space-time-stack
    run_one "costbrk-spacetimestack" "$REF_FVSOL" \
        "reference Cg+MG, space-time-stack region wall-time break-down (RESTART=$RESTART window)"
    use_kokkos_tool ""   # profiler back off for any later instrument
}

run_memhwm() {
    # Device high-water-mark: the reference peak the memory story is measured against.
    use_kokkos_tool memory-high-water-mark
    run_one "costbrk-memhwm" "$REF_FVSOL" \
        "reference Cg+MG, memory high-water-mark (RESTART=$RESTART window)"
    use_kokkos_tool ""
}

run_memtimeline() {
    # NF_MEM_SCOPE per-region device-pool timeline (MemoryProbe, independent of kokkos-tools). Forward
    # the env to every rank via kokkos_launch's MPIRUN_FORWARD_ENV hook; collect + plot the CSV after.
    use_kokkos_tool ""
    export NEOFOAM_MEM_TIMELINE=1 NEOFOAM_MEM_TIMELINE_FILE=memoryTimeline.csv
    export MPIRUN_FORWARD_ENV="NEOFOAM_MEM_TIMELINE NEOFOAM_MEM_TIMELINE_FILE"
    run_one "costbrk-memtimeline" "$REF_FVSOL" \
        "reference Cg+MG, NF_MEM_SCOPE device-pool timeline (RESTART=$RESTART window)"
    collect_mem_output "costbrk-memtimeline"
    unset NEOFOAM_MEM_TIMELINE NEOFOAM_MEM_TIMELINE_FILE MPIRUN_FORWARD_ENV
}

run_nsys() {
    # Opt-in Nsight Systems trace with NVTX (heavy). One .nsys-rep PER RANK.
    if [ "${NSYS:-0}" != "1" ]; then
        echo "   (nsys trace is opt-in; re-run as NSYS=1 ./phase1-paper-study-costbreakdown.sh nsys)"
        return
    fi
    if ! command -v nsys >/dev/null 2>&1; then
        echo "!! nsys not found on PATH -- skipping nsys trace"; return
    fi
    use_kokkos_tool ""   # kokkos-tools stdout profilers off; nsys captures the timeline
    # NVTX sources, forwarded to every rank:
    #   NEOFOAM_MEM_NVTX  -> each NF_MEM_SCOPE (the SIMPLE-loop phases: UEqn, pEqn assemble/solve,
    #                        turbulence, ...) emits an nvtxRangePush/Pop -> NVTX rows in the timeline.
    #   NEON_GINKGO_PROFILE-> GinkgoProfilingScope annotates the multigrid V-cycle (via Kokkos regions).
    export NEOFOAM_MEM_NVTX=1 NEON_GINKGO_PROFILE=1
    export MPIRUN_FORWARD_ENV="NEOFOAM_MEM_NVTX NEON_GINKGO_PROFILE"
    # Shorter window for the (large) per-rank traces; %q{OMPI_COMM_WORLD_RANK} splits output per rank.
    local nsteps="${NSYS_STEPS:-8}"
    pin_window "$RESTART" "$((RESTART + nsteps))"
    local out="$RESULTS/costbrk-nsys-$(date +%Y%m%d-%H%M%S)"
    export LAUNCH_WRAPPER="nsys profile --trace=cuda,nvtx,osrt --sample=process-tree --force-overwrite=true --cuda-memory-usage=true -o ${out}.rank%q{OMPI_COMM_WORLD_RANK}"
    run_one "costbrk-nsys" "$REF_FVSOL" \
        "reference Cg+MG, nsys+NVTX trace ($nsteps steps from restart $RESTART) -> ${out}.rank*.nsys-rep"
    unset LAUNCH_WRAPPER NEOFOAM_MEM_NVTX NEON_GINKGO_PROFILE MPIRUN_FORWARD_ENV
    pin_restart_window   # restore the full RESTART -> RESTART+STEPS window for any later instrument
    echo "   nsys reports: ${out}.rank{0..$((NP-1))}.nsys-rep  (Nsight Systems; NVTX rows = NF_MEM_SCOPE phases + Ginkgo V-cycle)"
}

collect_mem_output() {
    # Move the NF_MEM_SCOPE CSV next to the run log and plot the per-region ranking. No-op if the run
    # was skipped (no fresh CSV in cwd). $1 = run name (its log stem names the artifacts).
    local name="$1" log="${RUN_LOG[$name]}" stem
    [ -n "$log" ] || return 0
    stem="${log%.log}"
    [ -f memoryTimeline.csv ] || { echo "   (no memoryTimeline.csv produced -- skipped or probing off)"; return 0; }
    mv memoryTimeline.csv "${stem}.memoryTimeline.csv"
    if python3 "$MEMTOOLS/plot_memory_timeline.py" "${stem}.memoryTimeline.csv" \
            -o "${stem}.memoryTimeline.png" -n "$CELLS_PER_RANK" \
            > "${stem}.memoryTimeline.txt" 2>&1; then
        echo "   mem timeline -> ${stem}.memoryTimeline.{csv,png,txt}"
    else
        echo "   !! plot_memory_timeline.py failed (see ${stem}.memoryTimeline.txt)"
    fi
}

run_instrument() {
    case "$1" in
        spacetimestack|stack|space-time-stack) run_spacetimestack ;;
        memhwm|hwm|memory-high-water-mark)     run_memhwm ;;
        memtimeline|timeline)                  run_memtimeline ;;
        nsys)                                  run_nsys ;;
        *) echo "!! unknown instrument '$1' (spacetimestack|memhwm|memtimeline|nsys)" ;;
    esac
}

# --------------------------------------------------------------------- driver
INSTRUMENTS=("$@")
[ ${#INSTRUMENTS[@]} -eq 0 ] && INSTRUMENTS=(spacetimestack memhwm memtimeline)
for i in "${INSTRUMENTS[@]}"; do
    run_instrument "$i"
done

print_summary

echo
echo "cost-break-down artifacts under: $RESULTS/"
echo "  region wall-time  -> costbrk-spacetimestack-*.kokkos-profile.txt"
echo "  memory HWM        -> costbrk-memhwm-*.log"
echo "  memory timeline   -> costbrk-memtimeline-*.memoryTimeline.{csv,png,txt}"
echo "Next (AFTER profiling): ./phase2-paper-study-reference.sh   # clean, overhead-free timing"
