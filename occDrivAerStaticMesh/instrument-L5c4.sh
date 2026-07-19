#!/bin/bash
#SBATCH --job-name=instrument-L5c4
#SBATCH --nodelist=gpu-nvidia-h200-3
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00
#SBATCH --chdir=/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
#SBATCH --output=slurm-%x-%j.out
#
# Detailed cost break-down of the FASTEST cell from the levels×coarse sweep (§4.11e):
#   L5c4 = p-multigrid-mgsc-m3-L5-c4.json = Cg + global Multigrid + MG-level scale_correction=true
#          + pgmMerge3, max_levels=5, coarsest_solver max_iters=4, CACHED (cacheSolver=true, interval=0).
#
# The Phase-1 break-down (§4.1/§4.7) profiled the UNCACHED reference (p-multigrid.json), where
# momentumPredictor dominated (34%, uncached U-solver regen) and the pressure solve was dispatch-bound
# (GPU idle 78%, ~95k tiny kernels). L5c4 is a different regime: cached (no per-solve solver regen),
# MG-level sc, and a SHALLOW 5-level merge3 hierarchy (far fewer coarse levels → fewer V-cycle kernels).
# So its cost profile is genuinely new — this run finds L5c4's own bottleneck.
#
# Instruments (each a separate run, same config + same restart window, cached):
#   spacetimestack  KOKKOS_TOOL=space-time-stack -> per-region wall-time tree + GPU-vs-host split.
#                   THE primary artifact: momentumPredictor vs pEqn assemble vs pEqn solve vs turb vs
#                   halo exchange. -> <stem>.kokkos-profile.txt
#   memhwm          KOKKOS_TOOL=memory-high-water-mark -> device high-water-mark.
#   nsys            (opt-in, NSYS=1) LAUNCH_WRAPPER='nsys profile ...' -> kernel-level trace to confirm
#                   whether L5c4 is still dispatch-bound (kernel count, GPU-idle %) or now compute-bound.
#
# Usage:  ./instrument-L5c4.sh                 # spacetimestack + memhwm
#         ./instrument-L5c4.sh spacetimestack  # single instrument
#         NSYS=1 ./instrument-L5c4.sh nsys      # add the kernel-level nsys trace

STUDY_TYPE=l5c4-costbreakdown
STEPS="${STEPS:-30}"     # a modest window; region attributions are relative, not headline wall time
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

CFG="p-multigrid-mgsc-m3-L5-c4.json"

# Build the L5c4 fvSolution: swap the p configFile + pin caching ON (matches how the sweep ran it).
build_l5c4_fvsolution() {
    if [ ! -f "system/gko/$CFG" ]; then echo "!! missing system/gko/$CFG" >&2; return 1; fi
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$CFG;\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
    echo "$TMP_FVSOL"
}
FVSOL="$(build_l5c4_fvsolution)" || exit 1

run_spacetimestack() {
    use_kokkos_tool space-time-stack
    run_one "l5c4-spacetimestack" "$FVSOL" "L5c4 mgsc, space-time-stack region wall-time break-down"
    use_kokkos_tool ""
}
run_memhwm() {
    use_kokkos_tool memory-high-water-mark
    run_one "l5c4-memhwm" "$FVSOL" "L5c4 mgsc, device high-water-mark"
    use_kokkos_tool ""
}
run_nsys() {
    if ! command -v nsys >/dev/null 2>&1; then
        echo "!! nsys not on PATH (module load cuda) -- skipping nsys"; return
    fi
    local stem="$RESULTS/l5c4-nsys-$(date +%Y%m%d-%H%M%S)"
    # trace cuda+nvtx+mpi+osrt on all ranks; NVTX carries the NF_MEM_SCOPE / region ranges.
    export LAUNCH_WRAPPER="nsys profile --trace=cuda,nvtx,mpi,osrt --sample=none --cpuctxsw=none -f true -o ${stem}.rank%q{OMPI_COMM_WORLD_RANK}"
    run_one "l5c4-nsys" "$FVSOL" "L5c4 mgsc, nsys kernel-level trace (dispatch-bound check)"
    unset LAUNCH_WRAPPER
}
# Attributed trace: CPU sampling + backtraces on every CUDA synchronization API call, so the
# cudaDeviceSynchronize/cuStreamSynchronize mass (§4.11g/h) can be split by call stack (NeoN fence vs
# Ginkgo exec->synchronize vs libc). Heavier than run_nsys; rank 0 only via a wrapper is awkward, so
# trace all ranks and analyze rank0. Threshold keeps only syncs >5us (drops trivia).
run_nsysattr() {
    if ! command -v nsys >/dev/null 2>&1; then
        echo "!! nsys not on PATH -- skipping nsysattr"; return
    fi
    local stem="$RESULTS/l5c4-nsysattr-$(date +%Y%m%d-%H%M%S)"
    export LAUNCH_WRAPPER="nsys profile --trace=cuda,nvtx,mpi --cudabacktrace=sync:5000 --cpuctxsw=none -f true -o ${stem}.rank%q{OMPI_COMM_WORLD_RANK}"
    run_one "l5c4-nsysattr" "$FVSOL" "L5c4 mgsc, ATTRIBUTED nsys (sync backtraces): who calls cudaDeviceSynchronize?"
    unset LAUNCH_WRAPPER
    echo "   attributed reports: ${stem}.rank0.nsys-rep  (analyze: nsys stats --report cuda_api_sync <rep> ; or query the sqlite CUDA backtrace tables)"
    echo "   nsys reports: ${stem}.rank*.nsys-rep  (analyze: nsys stats --report cuda_gpu_kern_sum,cuda_api_sum <rep>)"
}

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=(spacetimestack memhwm)
echo "L5c4 cost break-down: $STEPS steps from restart $RESTART; instruments: ${SEL[*]}"
for inst in "${SEL[@]}"; do
    case "$inst" in
        spacetimestack) run_spacetimestack ;;
        memhwm)         run_memhwm ;;
        nsys)           run_nsys ;;
        nsysattr)       run_nsysattr ;;
        *) echo "!! unknown instrument '$inst' (spacetimestack memhwm nsys)";;
    esac
done

print_summary
echo
echo "primary artifact: $RESULTS/l5c4-spacetimestack-*.kokkos-profile.txt  (region wall-time tree + GPU/host split)"
