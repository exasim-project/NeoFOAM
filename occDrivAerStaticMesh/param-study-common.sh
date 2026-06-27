#!/bin/bash
#
# Shared infrastructure for the occDrivAre parameter studies. SOURCED by
# param-study.sh (linear-solver caseN comparison) and, via param-study-mg-common.sh, by
# param-study-mg.sh (Ginkgo multigrid headline comparison) and param-study-mg-tuning.sh
# (multigrid level/tuning sweep) — not run directly.
#
# Provides: environment + MPI/GPU binding, BIN/NP/STEPS/paths, controlDict
# run-window pinning with restore-on-exit, reset_to_t0, run_one, print_summary,
# and the RUNS[] accumulator that drives the summary table. Every run marches the
# identical 0 -> STEPS window from the same initial fields so comparisons are
# apples-to-apples. Mirrors run.sh for the environment.

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# ---------------------------------------------------------------- environment
module purge
module load gcc/13.3.0
module load cuda/13.0.2
module load cmake
module load openmpi
source "$HOME/OpenFOAM/openfoam/etc/bashrc"
export NEON_DEVICE=nvidia_h200
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Build type to take the neoSimpleFoam binary from. Default to the PROFILING (RelWithDebInfo)
# build: it runs every variant reliably. The PRODUCTION (Release/-O3) build SEGFAULTS during
# distributed Schwarz-preconditioner setup (before timestep 1) -- an optimization-sensitive bug
# (build types are the only CMake difference) -- so it is NOT the default despite being the only
# build with the NeoN_GINKGO_TAG (241deca) scale_correction config support the scalecorr variant
# needs. Override with NEON_BUILD=production once that Release-build crash is fixed.
NEON_BUILD="${NEON_BUILD:-profiling}"
BIN="/storage/home/greole/code/NeoFOAM/build/${NEON_BUILD}${NEON_DEVICE}/bin/neoSimpleFoam"
NP=4
STEPS=30
CFGDIR="system/paramStudy"
# Per-study results live under paramStudyResults/<type>/, one timestamped log per run:
# paramStudyResults/<STUDY_TYPE>/<run-name>-<YYYYmmdd-HHMMSS>.log. Each leaf study sets
# STUDY_TYPE before sourcing this file; "misc" is a fallback for direct/ad-hoc sourcing.
STUDY_TYPE="${STUDY_TYPE:-misc}"
RESULTS="paramStudyResults/${STUDY_TYPE}"
# Temp fvSolution written by the mg sweep (declared here so restore() cleans it
# up regardless of which study sourced this file).
TMP_FVSOL="$CFGDIR/.fvSolution.mgtmp"

mkdir -p "$RESULTS"

# ---------------------------------------------- optional Kokkos-Tools profiler
# Set KOKKOS_TOOL=<name> to load a kokkos-tools connector for every solver launch (param-study-best.sh,
# the mg sweeps, param-study-production.sh -- anything that runs through run_one or kokkos_launch).
# Off (empty) by default. The connectors are built into the active build's kokkos_tools_build when the
# binary was configured with NEOFOAM_ENABLE_KOKKOS_TOOLS=ON; names mirror profile-and-debug/
# run_with_kokkos_tool.sh. All profiler output is collected next to the run log under
# paramStudyResults/<type>/ by collect_kokkos_output: stdout tools (space-time-stack, the memory-*
# tools) land in the per-run <name>-<ts>.log, and the space-time-stack report is additionally copied
# to a standalone <name>-<ts>.kokkos-profile.txt; file tools (simple-kernel-timer .dat, perfetto/json)
# are moved to <name>-<ts>.<file>. (Sidecars are .txt/.dat, never .log, so param-study-table.sh's
# *.log glob ignores them.) Read a simple-kernel-timer .dat with the kp_reader next to the connector.
#   simple-kernel-timer  space-time-stack  memory-high-water-mark  memory-usage  memory-events
#   chrome-tracing  perfetto-connector  kernel-logger
KOKKOS_TOOL="${KOKKOS_TOOL:-}"
# Connector root: .../build/${NEON_BUILD}${NEON_DEVICE}/kokkos_tools_build (two dirs above BIN).
KTOOLS_ROOT="$(dirname "$(dirname "$BIN")")/kokkos_tools_build"

kokkos_tool_lib() {
    # Echo the connector .so path for tool name $1 (empty if unknown).
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

# Resolve KOKKOS_TOOLS_LIBS once (empty => profiler off / lib missing). The launchers forward it to
# the MPI ranks via `mpirun -x KOKKOS_TOOLS_LIBS` (see kokkos_launch).
KOKKOS_TOOLS_LIBS=""
if [ -n "$KOKKOS_TOOL" ]; then
    _ktlib="$(kokkos_tool_lib "$KOKKOS_TOOL")"
    if [ -n "$_ktlib" ] && [ -f "$_ktlib" ]; then
        KOKKOS_TOOLS_LIBS="$_ktlib"
        echo "kokkos-tools profiler ENABLED: $KOKKOS_TOOL -> $KOKKOS_TOOLS_LIBS"
    else
        echo "!! KOKKOS_TOOL='$KOKKOS_TOOL': no connector at '${_ktlib:-<unknown tool>}'"
        echo "   (rebuild with NEOFOAM_ENABLE_KOKKOS_TOOLS=ON, or pick a valid tool) -- running WITHOUT profiler"
    fi
fi
export KOKKOS_TOOLS_LIBS

kokkos_launch() {
    # mpirun wrapper that forwards KOKKOS_TOOLS_LIBS to the ranks when the profiler is enabled.
    # Usage: kokkos_launch <bin> [args...]   (adds -np $NP -parallel). Used by run_one and the
    # production launcher so both honour KOKKOS_TOOL.
    local bin="$1"; shift
    if [ -n "$KOKKOS_TOOLS_LIBS" ]; then
        mpirun -x KOKKOS_TOOLS_LIBS -np "$NP" "$bin" -parallel "$@"
    else
        mpirun -np "$NP" "$bin" -parallel "$@"
    fi
}

collect_kokkos_output() {
    # After a profiled run, gather the kokkos-tools output next to the run log so it follows the
    # paramStudyResults/<type>/<run-name>-<timestamp> layout. No-op unless a profiler is active.
    #   $1 = run log path   $2 = run cwd (default .)   $3 = pre-run marker file (optional)
    # - STDOUT connectors (space-time-stack, memory-hwm/usage, kernel-logger) print INTO the run log
    #   already; the self-contained space-time-stack report is ALSO copied to <stem>.kokkos-profile.txt
    #   so it is a standalone per-run artifact. NB: NOT a .log file -- param-study-table.sh globs
    #   $RESULTS/*.log and must not pick the profile up as a run.
    # - FILE connectors (simple-kernel-timer .dat, KOKKOS_PROFILE_EXPORT_JSON noname.json, perfetto
    #   traces) drop files in the run cwd; those are moved to <stem>.<file>. The marker (when given)
    #   scopes the move to files THIS run created, so a stray pre-existing .dat in the case dir is
    #   left alone.
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

# -------------------------------------------------- pin the run window to STEPS
# Back up the dictionaries we touch and restore them on exit (even on Ctrl-C).
# turbulenceProperties is included because the mg study's laminar variant toggles
# its simulationType; backing it up here keeps the case pristine even on interrupt.
cp system/controlDict           "system/controlDict.studybak"
cp system/fvSolution            "system/fvSolution.studybak"
cp constant/turbulenceProperties "constant/turbulenceProperties.studybak"
restore() {
    [ -f system/controlDict.studybak ] && mv system/controlDict.studybak system/controlDict
    [ -f system/fvSolution.studybak  ] && mv system/fvSolution.studybak  system/fvSolution
    [ -f constant/turbulenceProperties.studybak ] \
        && mv constant/turbulenceProperties.studybak constant/turbulenceProperties
    [ -f "$TMP_FVSOL" ] && rm -f "$TMP_FVSOL"
}
trap restore EXIT INT TERM

foamDictionary -entry startFrom     -set startTime -disableFunctionEntries system/controlDict >/dev/null
foamDictionary -entry startTime     -set 0         -disableFunctionEntries system/controlDict >/dev/null
foamDictionary -entry endTime       -set "$STEPS"  -disableFunctionEntries system/controlDict >/dev/null
foamDictionary -entry writeInterval -set "$STEPS"  -disableFunctionEntries system/controlDict >/dev/null

reset_to_t0() {
    # Drop every written time directory except 0 across all processors, so each
    # run marches the identical 0 -> STEPS window from the same initial fields.
    for p in processor*; do
        find "$p" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' ! -name '0' \
            -exec rm -rf {} + 2>/dev/null
    done
}

RUNS=()           # names of runs that actually executed, drives the summary table
declare -A RUN_LOG  # run name -> its actual (timestamped) log path, for print_summary/report_cache_reuse

WINDOW=10   # number of trailing log lines kept on screen during a run

tail_window() {
    # Live "last $WINDOW lines" display: while $1 (the solver pid) is alive, redraw
    # the tail of $2 (the log) in place, overwriting the previous block each tick so
    # only the most recent $WINDOW lines are ever shown. Falls back to a single tail
    # when stdout is not a terminal (no cursor control available).
    local pid="$1" log="$2" drawn=0 block
    if [ ! -t 1 ]; then
        wait "$pid" 2>/dev/null
        tail -n "$WINDOW" "$log" | sed 's/^/   | /'
        return
    fi
    printf '\033[?25l'   # hide cursor
    _draw() {
        [ "$drawn" -gt 0 ] && printf '\033[%dA' "$drawn"   # move up over old block
        printf '\033[J'                                    # clear from here down
        block=$(tail -n "$WINDOW" "$log" | sed 's/^/   | /')
        printf '%s\n' "$block"
        drawn=$(printf '%s\n' "$block" | wc -l)
    }
    while kill -0 "$pid" 2>/dev/null; do
        _draw
        sleep 0.5
    done
    _draw   # final redraw so the very last output is on screen
    printf '\033[?25h'   # show cursor
}

run_one() {
    # $1 = run name (log/summary key)   $2 = fvSolution to install   $3 = desc
    # $4 = solver binary (optional, defaults to $BIN / neoSimpleFoam) — set to e.g.
    #      "simpleFoam" for the native-OpenFOAM baseline run.
    local name="$1" src="$2" desc="$3" bin="${4:-$BIN}"
    # Resolve the per-run log path. On a gap-fill re-invocation reuse the newest existing
    # $RESULTS/<name>-<ts>.log for this run (so completed cases skip but still show in the
    # summary); otherwise mint a fresh timestamped path. FORCE=1 always writes a new log.
    # The trailing "-" before the timestamp anchors the glob so prefix names don't collide
    # (e.g. ...-rebuild10- never matches ...-rebuild100-).
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

    # Skip a case whose log already exists, so a re-invocation only fills in the
    # missing runs. The existing log is still added to RUNS so it appears in the
    # summary table. Override with FORCE=1 to re-run (writing a new timestamped log).
    if [ -z "${FORCE:-}" ] && [ -n "$existing" ]; then
        echo "   (skip: log exists — set FORCE=1 to re-run)"
        RUNS+=("$name")
        return
    fi

    cp "$src" system/fvSolution
    reset_to_t0

    local t0=$SECONDS
    : > "$log"
    # Marker to scope any kokkos-tools file output (e.g. simple-kernel-timer .dat) to THIS run when
    # collecting it afterwards. Created just before launch; empty/no-op when the profiler is off.
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
    printf "%-22s %6s %12s %14s %10s %12s\n" run steps "p_iters/it" "p_ms/solve" "cont" "exec_s"
    local c log steps cont exec_s pit pms
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
        printf "%-22s %6s %12s %14s %10s %12s\n" \
            "$c" "${steps:-0}" "${pit:--}" "${pms:--}" "${cont:--}" "${exec_s:--}"
    done
    echo "#########################################################"
    echo "per-run logs under: $RESULTS/"
}
