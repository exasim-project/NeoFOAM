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
STEPS=100
CFGDIR="system/paramStudy"
RESULTS="paramStudy/results"
# Temp fvSolution written by the mg sweep (declared here so restore() cleans it
# up regardless of which study sourced this file).
TMP_FVSOL="$CFGDIR/.fvSolution.mgtmp"

mkdir -p "$RESULTS"

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

RUNS=()   # names of runs that actually executed, drives the summary table

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
    local name="$1" src="$2" desc="$3" bin="${4:-$BIN}" log="$RESULTS/$1.log"
    echo "================================================================"
    echo " $name${desc:+ : $desc}"
    echo "   -> $log"
    echo "================================================================"

    # Skip a case whose log already exists, so a re-invocation only fills in the
    # missing runs. The existing log is still added to RUNS so it appears in the
    # summary table. Override with FORCE=1 to re-run (and overwrite) every case.
    if [ -z "${FORCE:-}" ] && [ -f "$log" ]; then
        echo "   (skip: log exists — set FORCE=1 to re-run)"
        RUNS+=("$name")
        return
    fi

    cp "$src" system/fvSolution
    reset_to_t0

    local t0=$SECONDS
    : > "$log"
    mpirun -np "$NP" "$bin" -parallel > "$log" 2>&1 &
    local pid=$!
    tail_window "$pid" "$log"
    wait "$pid"; local rc=$?
    echo "   exit=$rc  wall=$((SECONDS - t0))s  steps=$(grep -c '^Time = ' "$log")"
    RUNS+=("$name")
}

print_summary() {
    echo
    echo "######################## SUMMARY ########################"
    printf "%-22s %6s %12s %14s %10s %12s\n" run steps "p_iters/it" "p_ms/solve" "cont" "exec_s"
    local c log steps cont exec_s pit pms
    for c in "${RUNS[@]}"; do
        log="$RESULTS/$c.log"
        [ -f "$log" ] || continue
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
