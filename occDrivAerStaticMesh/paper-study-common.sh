#!/bin/bash
#
# Shared infrastructure for the OPTIMIZATION-PAPER parameter studies on occDrivAerStaticMesh.
# SOURCED by phase0-paper-study-spinup.sh (Phase 0), phase1-paper-study-costbreakdown.sh
# (Phase 1) and the later phaseN-paper-study-* wrappers -- not run directly.
#
# This is a FORK of param-study-common.sh, kept separate so the paper runs never mix with the
# earlier studies:
#   * all output goes to paperParamStudyResults/<STUDY_TYPE>/  (never paramStudyResults/)
#   * every measured run starts from the SEMI-CONVERGED RESTART field at iteration $RESTART
#     (produced once by phase0-paper-study-spinup.sh) instead of the uniform 0/ field, and marches
#     the identical RESTART -> RESTART+STEPS window so comparisons are apples-to-apples on a
#     representative operator/RHS.
#   * the original param-study-common.sh is left untouched, so the earlier studies are unaffected.
#
# Provides: environment + MPI/GPU binding, BIN/NP/RESTART/STEPS/paths, the reference-config
# builder (build_reference_fvsolution), restart-window pinning (pin_window / auto-pin),
# reset_to_restart, require_restart, run_one, print_summary, the kokkos-tools hooks
# (use_kokkos_tool / kokkos_launch / collect_kokkos_output) and the RUNS[] summary accumulator.

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# ---------------------------------------------------------------- environment
module purge
module load gcc/13.3.0
module load cuda/13.0.2
module load cmake
# Pinned, CUDA-aware OpenMPI (built --with-cuda): device-buffer MPI calls go GPU-direct.
module load openmpi-cuda/5.0.10
source "$HOME/OpenFOAM/openfoam/etc/bashrc"
export NEON_DEVICE=nvidia_h200
# One MPI rank per GPU. This node's 4 GPUs are indexed 0-3; the old value 1,2,3,4 referenced a
# nonexistent device 4, which silently stacked 2 ranks on GPU1 (GPU0 idle) -> oversubscription that
# inflated every measured timing (~1.9x s/step, up to ~4.3x p-solve; confirmed 50-step A/B 2026-07-17).
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Build to take neoSimpleFoam from. Default to the PROFILING (RelWithDebInfo) build: it runs every
# variant reliably (the PRODUCTION -O3 build still segfaults during distributed Schwarz setup).
NEON_BUILD="${NEON_BUILD:-profiling}"
BIN="/storage/home/greole/code/NeoFOAM/build/${NEON_BUILD}${NEON_DEVICE}/bin/neoSimpleFoam"
# Robust BIN resolution: fall back to the other h200 build if the configured one has no binary.
if [ ! -x "$BIN" ]; then
    for _alt in profiling production; do
        _altbin="/storage/home/greole/code/NeoFOAM/build/${_alt}${NEON_DEVICE}/bin/neoSimpleFoam"
        if [ "$_altbin" != "$BIN" ] && [ -x "$_altbin" ]; then
            echo "!! NEON_BUILD='$NEON_BUILD' has no binary ($BIN)" >&2
            echo "   -> falling back to the '$_alt' build: $_altbin" >&2
            NEON_BUILD="$_alt"; BIN="$_altbin"; break
        fi
    done
fi

NP=4
# The semi-converged restart iteration. FIXED at 1000 (not a swept parameter): every measured run
# starts from processor*/$RESTART/ and marches RESTART -> RESTART+STEPS.
RESTART=${RESTART:-1000}
STEPS="${STEPS:-30}"
CFGDIR="system/paramStudy"
# Per-study results live under paperParamStudyResults/<type>/ -- a DEDICATED root, never the
# earlier studies' paramStudyResults/. Each leaf study sets STUDY_TYPE before sourcing this file.
STUDY_TYPE="${STUDY_TYPE:-misc}"
RESULTS="paperParamStudyResults/${STUDY_TYPE}"
TMP_FVSOL="$CFGDIR/.fvSolution.papertmp"

mkdir -p "$RESULTS"

# ------------------------------------------------------- the reference config
# The paper's REFERENCE p-solver: fp64 PCG (Cg) + a GLOBAL (non-localized) Multigrid preconditioner,
# Pgm coarsening, max_levels=10, default smoother -- NO solver caching, NO scale-correction, NO
# mixed precision, NO PMIS. Identical to param-study-mg.sh's `base` variant. Both Phase 0 (spin-up)
# and Phase 1 (cost break-down) run exactly this config; only the instrumentation differs.
REF_CONFIG="${REF_CONFIG:-p-multigrid.json}"
MG_TEMPLATE="$CFGDIR/fvSolution.case3"

build_reference_fvsolution() {
    # Write the reference p-solver fvSolution into $TMP_FVSOL by swapping the p configFile path in
    # the case3 template (leaving its U/k/omega blocks -- the fixed baseline shared by every variant
    # -- untouched) and EXPLICITLY pinning solver caching OFF. Echoes the path.
    #
    # cacheSolver=false + preconditionerRebuildInterval=1 -> the Ginkgo solver regenerates a FRESH
    # Multigrid hierarchy every pressure solve instead of reusing a cached one via update_matrix_value.
    # This is deliberate on two counts:
    #   1. Correctness of the baseline: solver caching is one of the OPTIMIZATIONS Phase 3 evaluates,
    #      not part of the "CG + Multigrid without further optimizations" reference. Relying on the
    #      NeoN default is unsafe -- a stale incremental libNeoN build still defaults cacheSolver=ON
    #      (the committed source defaults it OFF), so the key must be pinned to state the intent.
    #   2. It fixes a real crash: with caching ON and rebuildInterval=0 (cache forever), the cached
    #      distributed-MG coarse operators drift out of sync with the evolving matrix over the
    #      spin-up transient and Ginkgo aborts mid-run (gko::DimensionMismatch in the V-cycle apply;
    #      the {rows,localNnz,nonLocalNnz} structure guard does not catch it). Regenerating each solve
    #      always matches the current matrix. rebuildInterval=1 is a backstop that forces the rebuild
    #      even if a stale binary ignores cacheSolver and keeps caching on.
    # The two keys are injected after the configFile line by sed (NOT foamDictionary, which
    # re-tokenises the "system/gko/..." path), matching param-study-mg-common.sh's cache runs.
    if [ ! -f "system/gko/$REF_CONFIG" ]; then
        echo "!! reference config system/gko/$REF_CONFIG missing" >&2; return 1
    fi
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$REF_CONFIG;\n        cacheSolver      false;\n        preconditionerRebuildInterval 1;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL" || return 1
    echo "$TMP_FVSOL"
}

# ---------------------------------------------- optional Kokkos-Tools profiler
# Set KOKKOS_TOOL=<name> (or call use_kokkos_tool <name> at runtime) to load a kokkos-tools connector
# for every solver launch. Off (empty) by default. Output is collected next to the run log by
# collect_kokkos_output. Names mirror the connectors built into the active build's kokkos_tools_build.
#   simple-kernel-timer  space-time-stack  memory-high-water-mark  memory-usage  memory-events
#   chrome-tracing  perfetto-connector  kernel-logger
KOKKOS_TOOL="${KOKKOS_TOOL:-}"
KTOOLS_ROOT="$(dirname "$(dirname "$BIN")")/kokkos_tools_build"

kokkos_tool_lib() {
    case "$1" in
        simple-kernel-timer)               echo "$KTOOLS_ROOT/profiling/simple-kernel-timer/libkp_kernel_timer.so" ;;
        space-time-stack)                  echo "$KTOOLS_ROOT/profiling/space-time-stack/libkp_space_time_stack.so" ;;
        memory-high-water-mark|memory-hwm) echo "$KTOOLS_ROOT/profiling/memory-hwm/libkp_hwm.so" ;;
        memory-usage)                      echo "$KTOOLS_ROOT/profiling/memory-usage/libkp_memory_usage.so" ;;
        memory-events)                     echo "$KTOOLS_ROOT/profiling/memory-events/libkp_memory_events.so" ;;
        chrome-tracing)                    echo "$KTOOLS_ROOT/profiling/chrome-tracing/libkp_chrome_tracing.so" ;;
        perfetto-connector)                echo "$KTOOLS_ROOT/profiling/perfetto-connector/libkp_perfetto_connector.so" ;;
        kernel-logger)                     echo "$KTOOLS_ROOT/debugging/kernel-logger/libkp_kernel_logger.so" ;;
        *)                                 echo "" ;;
    esac
}

use_kokkos_tool() {
    # Re-resolve KOKKOS_TOOLS_LIBS for tool name $1 at RUNTIME (empty $1 => profiler OFF). Lets a
    # single script cycle through several connectors (the Phase-1 cost break-down) without
    # re-sourcing this file. Loud, non-fatal warning if the connector is missing.
    KOKKOS_TOOL="$1"
    KOKKOS_TOOLS_LIBS=""
    if [ -n "$KOKKOS_TOOL" ]; then
        local lib; lib="$(kokkos_tool_lib "$KOKKOS_TOOL")"
        if [ -n "$lib" ] && [ -f "$lib" ]; then
            KOKKOS_TOOLS_LIBS="$lib"
            echo "kokkos-tools profiler ENABLED: $KOKKOS_TOOL -> $KOKKOS_TOOLS_LIBS"
        else
            echo "!! use_kokkos_tool '$KOKKOS_TOOL': no connector at '${lib:-<unknown tool>}'"
            echo "   (rebuild with NEOFOAM_ENABLE_KOKKOS_TOOLS=ON, or pick a valid tool) -- running WITHOUT profiler"
        fi
    fi
    export KOKKOS_TOOLS_LIBS
}

# Resolve the source-time KOKKOS_TOOL (if any) once.
KOKKOS_TOOLS_LIBS=""
use_kokkos_tool "$KOKKOS_TOOL"

kokkos_launch() {
    # mpirun wrapper that forwards KOKKOS_TOOLS_LIBS (and any names in MPIRUN_FORWARD_ENV) to the
    # ranks, and inserts LAUNCH_WRAPPER (e.g. an `nsys profile -o ...` prefix) before the binary.
    # Usage: kokkos_launch <bin> [args...]   (adds -np $NP -parallel).
    local bin="$1"; shift
    local xargs=()
    [ -n "$KOKKOS_TOOLS_LIBS" ] && xargs+=(-x KOKKOS_TOOLS_LIBS)
    local v
    for v in ${MPIRUN_FORWARD_ENV:-}; do
        [ -n "${!v+x}" ] && xargs+=(-x "$v")
    done
    # shellcheck disable=SC2086  # LAUNCH_WRAPPER is intentionally word-split into argv
    mpirun "${xargs[@]}" -np "$NP" ${LAUNCH_WRAPPER:-} "$bin" -parallel "$@"
}

collect_kokkos_output() {
    # Gather kokkos-tools output next to the run log. No-op unless a profiler is active.
    #   $1 = run log path   $2 = run cwd (default .)   $3 = pre-run marker file (optional)
    [ -n "$KOKKOS_TOOLS_LIBS" ] || return 0
    local log="$1" dir="${2:-.}" marker="$3" stem="${1%.log}" f base
    if grep -q 'BEGIN KOKKOS PROFILING REPORT' "$log" 2>/dev/null; then
        sed -n '/BEGIN KOKKOS PROFILING REPORT/,/END KOKKOS PROFILING REPORT/p' "$log" \
            > "${stem}.kokkos-profile.txt"
        echo "   kokkos-tools profile -> ${stem}.kokkos-profile.txt"
    fi
    while IFS= read -r -d '' f; do
        base="$(basename "$f")"
        mv "$f" "${stem}.${base}" && echo "   kokkos-tools output -> ${stem}.${base}"
    done < <(find "$dir" -maxdepth 1 -type f ${marker:+-newer "$marker"} \
        \( -name '*.dat' -o -name 'noname.json' -o -name '*.perfetto-trace' \) -print0 2>/dev/null)
}

# ------------------------------------------- pin the run window / restart guard
# Back up the dictionaries we touch and restore them on exit (even on Ctrl-C). Written time
# directories (including the $RESTART restart the spin-up produces) are NOT dictionaries and persist.
cp system/controlDict            "system/controlDict.paperbak"
cp system/fvSolution             "system/fvSolution.paperbak"
cp constant/turbulenceProperties "constant/turbulenceProperties.paperbak"
restore() {
    [ -f system/controlDict.paperbak ] && mv system/controlDict.paperbak system/controlDict
    [ -f system/fvSolution.paperbak  ] && mv system/fvSolution.paperbak  system/fvSolution
    [ -f constant/turbulenceProperties.paperbak ] \
        && mv constant/turbulenceProperties.paperbak constant/turbulenceProperties
    [ -f "$TMP_FVSOL" ] && rm -f "$TMP_FVSOL"
}
trap restore EXIT INT TERM

pin_window() {
    # Pin controlDict to the window [$1, $2], writing ONLY the final time ($2). $1 is read as the
    # start field, so it must exist across processor*/. Used with (0, RESTART) by the spin-up and
    # (RESTART, RESTART+STEPS) by every measured run.
    local start="$1" end="$2"
    foamDictionary -entry startFrom     -set startTime        -disableFunctionEntries system/controlDict >/dev/null
    foamDictionary -entry startTime     -set "$start"         -disableFunctionEntries system/controlDict >/dev/null
    foamDictionary -entry stopAt        -set endTime          -disableFunctionEntries system/controlDict >/dev/null
    foamDictionary -entry endTime       -set "$end"           -disableFunctionEntries system/controlDict >/dev/null
    foamDictionary -entry writeInterval -set "$((end - start))" -disableFunctionEntries system/controlDict >/dev/null
}
pin_restart_window() { pin_window "$RESTART" "$((RESTART + STEPS))"; }

# Measured studies get the restart window pinned automatically at source (mirrors the original
# common's behaviour). The spin-up sets PAPER_SKIP_WINDOW_PIN=1 before sourcing and pins its own
# 0 -> RESTART window, since the restart does not exist yet.
[ -z "${PAPER_SKIP_WINDOW_PIN:-}" ] && pin_restart_window

reset_times() {
    # Drop every written time directory EXCEPT the ones named in $@, across all processors.
    local p k keepargs=()
    for k in "$@"; do keepargs+=( ! -name "$k" ); done
    for p in processor*; do
        find "$p" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' "${keepargs[@]}" \
            -exec rm -rf {} + 2>/dev/null
    done
}
reset_to_restart() {
    # Keep 0/ (provenance) and $RESTART/ (the restart field); drop everything else, so each variant
    # marches the identical RESTART -> RESTART+STEPS window from the frozen restart.
    reset_times 0 "$RESTART"
}

require_restart() {
    # Abort with a clear message if the restart field is missing. Measured studies call this before
    # any run so a stale/absent restart fails loudly instead of silently starting from 0/.
    if [ ! -d "processor0/$RESTART" ]; then
        echo "!! restart field processor0/$RESTART not found." >&2
        echo "   Run ./phase0-paper-study-spinup.sh first to generate the iteration-$RESTART restart." >&2
        exit 1
    fi
}

RUNS=()             # names of runs that actually executed, drives the summary table
declare -A RUN_LOG  # run name -> its actual (timestamped) log path

WINDOW=10   # number of trailing log lines kept on screen during a run

tail_window() {
    # Live "last $WINDOW lines" display while $1 (the solver pid) is alive; falls back to a single
    # tail when stdout is not a terminal.
    local pid="$1" log="$2" drawn=0 block
    if [ ! -t 1 ]; then
        wait "$pid" 2>/dev/null
        tail -n "$WINDOW" "$log" | sed 's/^/   | /'
        return
    fi
    printf '\033[?25l'
    _draw() {
        [ "$drawn" -gt 0 ] && printf '\033[%dA' "$drawn"
        printf '\033[J'
        block=$(tail -n "$WINDOW" "$log" | sed 's/^/   | /')
        printf '%s\n' "$block"
        drawn=$(printf '%s\n' "$block" | wc -l)
    }
    while kill -0 "$pid" 2>/dev/null; do
        _draw
        sleep 0.5
    done
    _draw
    printf '\033[?25h'
}

run_one() {
    # $1 = run name (log/summary key)   $2 = fvSolution to install   $3 = desc
    # $4 = solver binary (optional, defaults to $BIN / neoSimpleFoam).
    local name="$1" src="$2" desc="$3" bin="${4:-$BIN}"
    local existing log
    existing=$(ls -1t "$RESULTS/${name}"-*.log 2>/dev/null | head -1)
    if [ -n "${FORCE:-}" ] || [ -z "$existing" ]; then
        log="$RESULTS/${name}-$(date +%Y%m%d-%H%M%S).log"
    else
        log="$existing"
    fi
    RUN_LOG["$name"]="$log"
    echo "================================================================"
    echo " $name${desc:+ : $desc}"
    echo "   -> $log"
    echo "================================================================"

    if [ -z "${FORCE:-}" ] && [ -n "$existing" ]; then
        echo "   (skip: log exists — set FORCE=1 to re-run)"
        RUNS+=("$name")
        return
    fi

    if [ ! -x "$bin" ]; then
        echo "   !! solver binary not found/executable: $bin"
        echo "      build it (or set NEON_BUILD to a built variant) -- skipping '$name'"
        return 1
    fi

    cp "$src" system/fvSolution
    reset_to_restart

    local t0=$SECONDS
    : > "$log"
    local ktmark=""
    [ -n "$KOKKOS_TOOLS_LIBS" ] && ktmark="$(mktemp)"
    kokkos_launch "$bin" > "$log" 2>&1 &
    local pid=$!
    tail_window "$pid" "$log"
    wait "$pid"; local rc=$?
    collect_kokkos_output "$log" "." "$ktmark"
    [ -n "$ktmark" ] && rm -f "$ktmark"
    echo "   exit=$rc  wall=$((SECONDS - t0))s  steps=$(grep -c '^Time = ' "$log")"
    RUNS+=("$name")
}

print_summary() {
    echo
    echo "######################## SUMMARY ########################"
    printf "%-26s %6s %12s %14s %10s %12s %10s\n" \
        run steps "p_iters/it" "p_ms/solve" "cont" "exec_s" "s/step"
    local c log steps cont exec_s pit pms sstep
    for c in "${RUNS[@]}"; do
        log="${RUN_LOG[$c]}"
        [ -n "$log" ] && [ -f "$log" ] || continue
        steps=$(grep -c '^Time = ' "$log")
        cont=$(grep 'sum local' "$log" | tail -1 | sed -E 's/.*sum local = ([0-9.eE+-]+),.*/\1/')
        exec_s=$(grep 'ExecutionTime' "$log" | tail -1 | sed -E 's/.*ExecutionTime = ([0-9.]+) s.*/\1/')
        read -r pit pms < <(awk '
            /Solving for p,/ {
                if (match($0, /No Iterations [0-9]+/)) { sit += substr($0, RSTART+14, RLENGTH-14) }
                if (match($0, /Solve time = [0-9.]+/))  { sst += substr($0, RSTART+13, RLENGTH-13) }
                n++
            }
            END { if (n) printf "%.0f %.0f", sit/n, sst/n; else printf "- -" }' "$log")
        # Overhead-free time-per-timestep = total ExecutionTime / SIMPLE iterations in the window.
        if [ -n "$exec_s" ] && [ "${steps:-0}" -gt 0 ]; then
            sstep=$(awk "BEGIN{printf \"%.3f\", $exec_s/$steps}")
        else
            sstep="-"
        fi
        printf "%-26s %6s %12s %14s %10s %12s %10s\n" \
            "$c" "${steps:-0}" "${pit:--}" "${pms:--}" "${cont:--}" "${exec_s:--}" "$sstep"
    done
    echo "#########################################################"
    echo "per-run logs under: $RESULTS/"
}
