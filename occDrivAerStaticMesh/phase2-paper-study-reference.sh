#!/bin/bash
#SBATCH --job-name=phase2-paper-reference
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Submit:      sbatch phase2-paper-study-reference.sh
# Or run live: salloc -w gpu-nvidia-h200-3 -n32 -t 08:00:00 --gres gpu:4   then   ./phase2-paper-study-reference.sh
# Override repeat count: N=5 ./phase2-paper-study-reference.sh
#
# Optimization-paper study -- PHASE 2: clean reference performance (overhead-free timing).
#
# The paper's HEADLINE number and the denominator every speedup is measured against. Runs the SAME
# reference solver as Phase 1 (fp64 Cg + global Multigrid, p-multigrid.json, cacheSolver off) from the
# SAME iteration-$RESTART restart window, but with EVERY instrument OFF -- no kokkos-tools connector,
# no NF_MEM_SCOPE timeline/NVTX, no nsys, no profiling env -- so the measured wall time carries zero
# profiling overhead. Run AFTER Phase 1 (which collects the instrumented cost break-down): the
# space-time-stack / nsys traces perturb per-timestep cost, so the true time-per-timestep is measured
# separately here.
#
# Primary metric: time per timestep = ExecutionTime / SIMPLE iterations in the window. Repeated N (=3)
# times; mean +/- spread is reported, since GPU/host timing on this host-bound case is noisy shot to
# shot. This clean s/step -- NOT any Phase-1 instrumented wall time -- is the speedup denominator.
#
# Requires the Phase-0 restart (processor*/$RESTART/); aborts loudly if it is missing.

STUDY_TYPE=reference
# sbatch copies the script into a spool dir, so $0/BASH_SOURCE point there, not at the
# real script. Fall back to $PWD (guaranteed correct by #SBATCH --chdir) when the sibling
# common script isn't next to us.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"

require_restart

# Assert ZERO instrumentation. Any of these set would taint the headline timing; unset defensively and
# warn (so a stray env var from a previous phase in the same shell can't silently perturb the number).
use_kokkos_tool ""   # no kokkos-tools connector (also clears KOKKOS_TOOLS_LIBS)
for _v in NEOFOAM_MEM_TIMELINE NEOFOAM_MEM_TIMELINE_FILE NEOFOAM_MEM_NVTX NEOFOAM_MEM_ALLOC_RECORDS \
          NEON_GINKGO_PROFILE KOKKOS_TOOL KOKKOS_TOOLS_LIBS LAUNCH_WRAPPER MPIRUN_FORWARD_ENV \
          KOKKOS_PROFILE_LIBRARY; do
    if [ -n "${!_v:-}" ]; then
        echo "!! $_v is set ('${!_v}') -- Phase 2 must be overhead-free; unsetting for the clean run"
        unset "$_v"
    fi
done

REF_FVSOL="$(build_reference_fvsolution)" || { echo "!! could not build reference fvSolution"; exit 1; }

N="${N:-3}"
echo "Phase 2: $N clean reference runs, window $RESTART -> $((RESTART + STEPS)), no instrumentation"
for i in $(seq 1 "$N"); do
    run_one "reference-run${i}" "$REF_FVSOL" \
        "clean reference Cg+MG timing, run ${i}/${N} (restart $RESTART, overhead-free)"
done

print_summary

# ------------------------------------------------ headline: time per timestep (mean +/- spread)
report_timing() {
    echo
    echo "################## HEADLINE: time per timestep (overhead-free) ##################"
    printf "%-18s %8s %12s %12s\n" run steps "exec_s" "s/step"
    local c log steps exec_s sstep
    local vals=()
    for c in "${RUNS[@]}"; do
        log="${RUN_LOG[$c]}"
        [ -n "$log" ] && [ -f "$log" ] || continue
        steps=$(grep -c '^Time = ' "$log")
        exec_s=$(grep 'ExecutionTime' "$log" | tail -1 | sed -E 's/.*ExecutionTime = ([0-9.]+) s.*/\1/')
        [ -n "$exec_s" ] && [ "${steps:-0}" -gt 0 ] || continue
        sstep=$(awk "BEGIN{printf \"%.4f\", $exec_s/$steps}")
        vals+=("$sstep")
        printf "%-18s %8s %12s %12s\n" "$c" "$steps" "$exec_s" "$sstep"
    done
    if [ ${#vals[@]} -gt 0 ]; then
        printf '%s\n' "${vals[@]}" | awk '
            { x[NR]=$1; s+=$1; if(NR==1||$1<mn)mn=$1; if(NR==1||$1>mx)mx=$1 }
            END{
                m=s/NR;
                for(i=1;i<=NR;i++){d=x[i]-m; v+=d*d}
                sd=(NR>1)?sqrt(v/(NR-1)):0;
                printf "--------------------------------------------------------------\n";
                printf "mean s/step = %.4f  +/- %.4f (sd, n=%d)   min %.4f  max %.4f\n", m, sd, NR, mn, mx;
            }'
    fi
    echo "################################################################################"
    echo "This mean s/step is the paper's speedup DENOMINATOR (reference Cg+MG, no optimization)."
}
report_timing

echo
echo "clean-timing logs under: $RESULTS/"
