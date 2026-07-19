#!/bin/bash
#
# High-water-mark DEVICE-MEMORY measurement for the 4 red-boxed champions of the merge-levels sweep
# (paperParamStudyResults/merge-sweep-pms.png). One best cell per quadrant:
#
#   global-nosc   Global MG, no scale correction   L2, merge3   (763 ms in the sweep)
#   global-scpost Global MG, sc post-pass          L4, merge2   (782 ms)
#   loc-nosc      Localized MG, no scale correction L6, merge3   (733 ms)
#   loc-scpost    Localized MG, sc post-pass        L8, merge2   (756 ms)
#
# Each cell is (re)run for STEPS steps through the SAME phase3h/phase3i harness that produced the
# sweep -- so the config is byte-identical to the plotted cell -- while an nvidia-smi sampler records
# peak memory.used on every GPU (0.5 s cadence). Peak - baseline = the config's device footprint
# above the fixed ~38 GB Umpire pool (= the Ginkgo MG hierarchy, allocated by gko::CudaAllocator
# OUTSIDE the pool; see the header of measure-solver-memory.sh). nvidia-smi captures pool + raw both.
#
# Runs go to a DEDICATED -memtest results dir (STUDY_SUFFIX) with FORCE=1 so the real 50-step sweep
# logs the plot reads are NOT overwritten with short runs.
#
# Usage:  ./measure-merge-best-memory.sh                 # all 4
#         ./measure-merge-best-memory.sh loc-nosc        # subset
#         STEPS=15 ./measure-merge-best-memory.sh

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

STEPS="${STEPS:-20}"
SUFFIX="-memtest"
MEMDIR="paperParamStudyResults/merge-best-memory"
mkdir -p "$MEMDIR"

# name -> "phase_script | env-prefix | cell | human description"
declare -A CELL=(
    [global-nosc]="phase3h-paper-study-mgsc-coarse-reltol.sh | MERGE=3 SC=false LEVELS=2 | L2e1 | Global MG, no sc, L2 merge3"
    [global-scpost]="phase3h-paper-study-mgsc-coarse-reltol.sh | MERGE=2 SC=post LEVELS=4 | L4e1 | Global MG, sc post-pass, L4 merge2"
    [loc-nosc]="phase3i-paper-study-localized-reltol.sh | MERGE=3 SC=none LEVELS=6 | L6e1 | Localized MG, no sc, L6 merge3"
    [loc-scpost]="phase3i-paper-study-localized-reltol.sh | MERGE=2 SC=post LEVELS=8 | L8e1 | Localized MG, sc post-pass, L8 merge2"
)
ORDER=(global-nosc global-scpost loc-nosc loc-scpost)

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")

echo "################################################################"
echo " Merge-sweep champions -- peak device-memory (nvidia-smi), $STEPS steps each"
echo " cells: ${SEL[*]}"
echo "################################################################"

SUMMARY=()

for name in "${SEL[@]}"; do
    spec="${CELL[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown cell '$name' (${ORDER[*]})"; continue; }
    script="$(echo "$spec" | awk -F' \\| ' '{print $1}')"
    envp="$(echo "$spec"   | awk -F' \\| ' '{print $2}')"
    cell="$(echo "$spec"   | awk -F' \\| ' '{print $3}')"
    desc="$(echo "$spec"   | awk -F' \\| ' '{print $4}')"

    stamp="$(python3 -c 'import time; print(time.strftime("%Y%m%d-%H%M%S"))')"
    smi="$MEMDIR/${name}-nvsmi-${stamp}.csv"
    baseline="$MEMDIR/${name}-baseline-${stamp}.txt"

    echo
    echo "=== $name : $desc ==="
    echo "    $envp ./$script $cell  (STEPS=$STEPS, dir suffix $SUFFIX)"

    # settle, record baseline (memory.used before the run)
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$baseline" 2>/dev/null

    # background sampler: every 0.5 s, all GPUs
    ( while true; do
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
        echo "---"
        sleep 0.5
      done ) > "$smi" 2>/dev/null &
    smipid=$!

    # run the single champion cell through its own phase harness
    env $envp STEPS="$STEPS" TOLS="0.1" STUDY_SUFFIX="$SUFFIX" FORCE=1 \
        bash "./$script" "$cell" > "$MEMDIR/${name}-run-${stamp}.out" 2>&1
    rc=$?

    kill "$smipid" 2>/dev/null; wait "$smipid" 2>/dev/null

    # per-GPU peak and baseline; report max-across-GPUs peak and delta
    read -r peakmax deltamax < <(awk -F', *' -v base="$baseline" '
        BEGIN{ while((getline l < base)>0){ split(l,a,/, */); b[a[1]]=a[2] } }
        /^[0-9]/{ if($2>mx[$1]) mx[$1]=$2 }
        END{
            pmax=0; dmax=0;
            for(i in mx){
                printf "      GPU%s peak=%d MiB  baseline=%d MiB  delta=%+d MiB\n", i, mx[i], b[i]+0, mx[i]-(b[i]+0) > "/dev/stderr"
                if(mx[i]>pmax) pmax=mx[i];
                d=mx[i]-(b[i]+0); if(d>dmax) dmax=d;
            }
            printf "%d %d\n", pmax, dmax
        }' "$smi")

    steps_done=$(grep -c '^Time = ' "$MEMDIR/${name}-run-${stamp}.out" 2>/dev/null)
    pms=$(awk '/Solving for p,/ && match($0,/Solve time = [0-9.]+/){ s+=substr($0,RSTART+13,RLENGTH-13); n++ } END{ if(n) printf "%.0f", 1000*s/n; else print "-" }' "$MEMDIR/${name}-run-${stamp}.out")
    echo "    exit=$rc  steps=$steps_done  mean p-solve=${pms} ms"
    echo "    peak(max GPU)=${peakmax} MiB   delta over baseline=${deltamax} MiB"
    echo "    sampler: $smi"
    SUMMARY+=("$(printf '%-14s %-34s peak=%6s MiB  delta=%6s MiB  psolve=%5s ms  steps=%s' "$name" "$desc" "${peakmax:--}" "${deltamax:--}" "${pms:--}" "${steps_done:-0}")")
done

echo
echo "######################## HIGH-WATER-MARK SUMMARY ########################"
printf '%s\n' "${SUMMARY[@]}"
echo "########################################################################"
echo "delta = peak device memory.used above pre-run baseline (the solver-hierarchy footprint)."
echo "csv + run logs under: $MEMDIR/"
