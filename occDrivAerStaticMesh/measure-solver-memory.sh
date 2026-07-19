#!/bin/bash
# Measure PEAK device memory (nvidia-smi) for the best global-mgsc cell vs the best localized MG,
# on the CURRENT build, same session. The NeoN [mem] DEVICE_POOL probe cannot compare these: NeoN
# fields use the Umpire pool (allocator=UmpirePool, 38 GB reserved, config-independent) while the
# Ginkgo MG hierarchy is allocated by gko::CudaAllocator (raw cudaMalloc, OUTSIDE the pool) -- see
# ginkgo bridge createGkoExecutor -> ext::kokkos::create_executor -> CudaAllocator (spaces.hpp:229).
# nvidia-smi memory.used captures BOTH (pool + Ginkgo raw), so the per-config DELTA above the fixed
# 38 GB pool is the solver-hierarchy footprint.
#
#   L4c8-mem  p-multigrid-mgsc-m3-L4-c8.json   global Cg+MG + MG-level sc + pgmMerge3 (sweep best-ish)
#   loc-mem   p-multigrid-localized-solver.json localized Schwarz{Multigrid}, plain Pgm (best practice)
#
# Short window (STEPS=15) is enough: the MG hierarchy is built on solve 1 and cached, so peak memory
# is reached within a few steps. nvidia-smi samples all 4 GPUs every 0.5 s during each run; we report
# the max memory.used per GPU (they are ~symmetric under the 4-way decomposition).

STUDY_TYPE=solver-memory
STEPS="${STEPS:-15}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -f "$SELF_DIR/paper-study-common.sh" ] || SELF_DIR="$PWD"
source "$SELF_DIR/paper-study-common.sh"
require_restart

build_variant_fvsolution() {
    local cfg="$1" cache="true" interval="0"
    if [ ! -f "system/gko/$cfg" ]; then echo "!! missing system/gko/$cfg" >&2; return 1; fi
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$cfg;\n        cacheSolver      $cache;\n        preconditionerRebuildInterval $interval;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
}

# $1 = name, $2 = config, $3 = desc. Wraps run_one with an nvidia-smi peak-memory sampler.
run_mem() {
    local name="$1" cfg="$2" desc="$3"
    build_variant_fvsolution "$cfg" || { echo "   skip $name"; return; }
    local smi="$RESULTS/${name}-nvsmi-$(date +%Y%m%d-%H%M%S).csv"
    # baseline (pre-run) memory so we can subtract other tenants if any
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$RESULTS/${name}-baseline.txt" 2>/dev/null
    # start sampler (all GPUs, 0.5 s), stop it after the run
    ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; echo "---"; sleep 0.5; done ) > "$smi" 2>/dev/null &
    local smipid=$!
    run_one "$name" "$TMP_FVSOL" "$desc"
    kill "$smipid" 2>/dev/null; wait "$smipid" 2>/dev/null
    # peak memory.used per GPU index from the sampler
    echo "   [mem] peak device memory.used (MiB) during $name:"
    awk -F', *' '/^[0-9]/{ if($2>mx[$1])mx[$1]=$2 } END{ for(i=0;i<=3;i++) if(i in mx) printf "      GPU%d peak=%d MiB\n", i, mx[i] }' "$smi"
    echo "   [mem] sampler csv: $smi"
}

declare -A VARIANT=(
    [L4c8-mem]="p-multigrid-mgsc-m3-L4-c8.json | GLOBAL Cg+MG + MG-level sc + pgmMerge3 (L4,c8)"
    [loc-mem]="p-multigrid-localized-solver.json | LOCALIZED Schwarz{Multigrid}, plain Pgm (best practice)"
)
ORDER=(L4c8-mem loc-mem)

SEL=("$@"); [ ${#SEL[@]} -eq 0 ] && SEL=("${ORDER[@]}")
echo "solver-memory: $STEPS steps/variant from restart $RESTART; variants: ${SEL[*]}"
for name in "${SEL[@]}"; do
    spec="${VARIANT[$name]:-}"
    [ -n "$spec" ] || { echo "!! unknown variant '$name' (${ORDER[*]})"; continue; }
    cfg="${spec%% *}"; desc="${spec#*| }"
    run_mem "$name" "$cfg" "$desc"
done

print_summary
echo
echo "COMPARE: peak memory.used delta above the fixed ~38 GB Umpire pool = Ginkgo MG-hierarchy footprint."
echo "logs+csv under: $RESULTS/"
