#!/bin/bash
#
# CLEAN A/B: is the merge-sweep's absolute pressure-solve time inflated by GPU oversubscription?
#
# Same cell, SAME 50-step restart window, SAME build/config -- the ONLY difference is line 32 of
# paper-study-common.sh:
#     A (bind1234)  CUDA_VISIBLE_DEVICES=1,2,3,4   the original: device "4" invalid on a 0-3 node
#                                                  -> 2 ranks stack on GPU1, GPU0 idle (oversubscribed)
#     B (bind0123)  CUDA_VISIBLE_DEVICES=0,1,2,3   one MPI rank per GPU (no stacking)
#
# Two cells, both no-sc (the merge-sweep champions of their branch):
#     gnosc  global no-sc  L2 merge3   (plotted 763 ms; memtest@bind0123 ~173 ms)
#     lnosc  localized no-sc L6 merge3 (plotted 733 ms; the overall sweep best)
#
# Each run: nvidia-smi sampler (0.5 s) confirms the bind pattern (A: GPU1 doubled, GPU0 idle;
# B: 4 GPUs symmetric). Output to paperParamStudyResults/<study>-ab<A|B>/ with FORCE=1 so the real
# sweep logs are untouched. common.sh is patched in place and RESTORED at the end (and on Ctrl-C).

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

STEPS="${STEPS:-50}"
COMMON="paper-study-common.sh"
BK="/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/6e9c8f01-a586-4004-9e1e-00415f6451b2/scratchpad/common-ab-backup.sh"
MEMDIR="paperParamStudyResults/ab-binding"
mkdir -p "$MEMDIR"

cp -p "$COMMON" "$BK"
restore_common() { cp -p "$BK" "$COMMON"; echo "[restored $COMMON]"; }
trap restore_common EXIT INT TERM

set_bind() {   # $1 = "1,2,3,4" or "0,1,2,3"
    sed -i -E "s/^export CUDA_VISIBLE_DEVICES=[0-9,]+.*/export CUDA_VISIBLE_DEVICES=$1/" "$COMMON"
    local got; got=$(grep -E '^export CUDA_VISIBLE_DEVICES=' "$COMMON")
    echo "   [bind] $got"
}

# name -> "script | env | cell | suffix-stem | desc"
declare -A CELL=(
    [gnosc]="phase3h-paper-study-mgsc-coarse-reltol.sh | MERGE=3 SC=false LEVELS=2 | L2e1 | mgnosc-coarse-reltol | Global no-sc L2 merge3"
    [lnosc]="phase3i-paper-study-localized-reltol.sh | MERGE=3 SC=none LEVELS=6 | L6e1 | localized-coarse-reltol | Localized no-sc L6 merge3"
)
CELLORDER=(gnosc lnosc)

# binding label -> value
declare -A BIND=( [A]="1,2,3,4" [B]="0,1,2,3" )

SUMMARY=()

run_cell() {   # $1 = cell key, $2 = bind label (A|B)
    local key="$1" bl="$2" spec script envp cell stem desc
    spec="${CELL[$key]}"
    script="$(echo "$spec" | awk -F' \\| ' '{print $1}')"
    envp="$(echo "$spec"   | awk -F' \\| ' '{print $2}')"
    cell="$(echo "$spec"   | awk -F' \\| ' '{print $3}')"
    stem="$(echo "$spec"   | awk -F' \\| ' '{print $4}')"
    desc="$(echo "$spec"   | awk -F' \\| ' '{print $5}')"
    local suffix="-ab${bl}"
    local stamp; stamp="$(python3 -c 'import time;print(time.strftime("%Y%m%d-%H%M%S"))')"
    local smi="$MEMDIR/${key}-${bl}-nvsmi-${stamp}.csv"

    echo
    echo "=== ${key} / bind $bl (${BIND[$bl]}) : $desc ==="
    set_bind "${BIND[$bl]}"

    ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; echo "---"; sleep 0.5; done ) > "$smi" 2>/dev/null &
    local smipid=$!
    env $envp STEPS="$STEPS" TOLS="0.1" STUDY_SUFFIX="$suffix" FORCE=1 \
        bash "./$script" "$cell" > "$MEMDIR/${key}-${bl}-run-${stamp}.out" 2>&1
    kill "$smipid" 2>/dev/null; wait "$smipid" 2>/dev/null

    local log; log=$(ls -1t "paperParamStudyResults/${stem}${suffix}/${cell}"-2026*.log 2>/dev/null | head -1)
    local steps pms pit
    steps=$(grep -c '^Time = ' "$log" 2>/dev/null)
    read pit pms < <(awk '/Solving for p,/{ if(match($0,/No Iterations [0-9]+/)){si+=substr($0,RSTART+14,RLENGTH-14)} if(match($0,/Solve time = [0-9.]+/)){ss+=substr($0,RSTART+13,RLENGTH-13)} n++ } END{ if(n) printf "%.1f %.1f", si/n, ss/n }' "$log")
    # skip the first (hierarchy-build) solve for a steady-state mean too
    local pms_steady
    pms_steady=$(awk '/Solving for p,/{ if(match($0,/Solve time = [0-9.]+/)){ n++; if(n>1){ss+=substr($0,RSTART+13,RLENGTH-13); m++} } } END{ if(m) printf "%.1f", ss/m }' "$log")
    local execs; execs=$(grep 'ExecutionTime' "$log" | tail -1 | sed -E 's/.*ExecutionTime = ([0-9.]+) s.*/\1/')
    local sstep; sstep=$(awk "BEGIN{ if(\"$steps\"+0>0) printf \"%.3f\", $execs/$steps; else print \"-\" }")
    # per-GPU peak (shows the bind pattern)
    local peaks; peaks=$(awk -F', *' '/^[0-9]/{ if($2>mx[$1])mx[$1]=$2 } END{ for(i=0;i<=3;i++) printf "%d ", mx[i] }' "$smi")

    echo "   steps=$steps  iters=$pit  p_ms(all)=$pms  p_ms(steady)=$pms_steady  s/step=$sstep"
    echo "   per-GPU peak MiB: $peaks   sampler=$smi"
    SUMMARY+=("$(printf '%-6s bind%s  iters=%-5s p_ms_all=%-7s p_ms_steady=%-7s s/step=%-6s  GPUpeaks=[%s]' "$key" "$bl" "${pit:--}" "${pms:--}" "${pms_steady:--}" "${sstep:--}" "$peaks")")
}

echo "################################################################"
echo " A/B GPU-binding confirmation: $STEPS-step window, cells: ${CELLORDER[*]}"
echo " A = CUDA_VISIBLE_DEVICES=1,2,3,4 (stacked, original)"
echo " B = CUDA_VISIBLE_DEVICES=0,1,2,3 (one rank/GPU)"
echo "################################################################"

for key in "${CELLORDER[@]}"; do
    run_cell "$key" A
    run_cell "$key" B
done

echo
echo "############################ A/B SUMMARY ############################"
printf '%s\n' "${SUMMARY[@]}"
echo "####################################################################"
echo "logs + csv: $MEMDIR/   (common.sh restored)"
