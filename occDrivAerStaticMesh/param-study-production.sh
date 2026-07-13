#!/bin/bash
#
# PRODUCTION full run for occDrivAreStaticMesh.
#
# Same selected best-practice solver setup as param-study-best.sh -- the pMG-localized-cache-
# rebuild100 configuration (Ginkgo Cg + LOCALIZED Multigrid preconditioner, max_levels=10, solver
# cache ON, l1ScaledResidual stop, fp64) -- but marched over a FULL production window instead of the
# short STEPS comparison window the param-study harness pins by default:
#
#   stopAt          endTime;
#   endTime         4000;          (with deltaT=1 -> 4000 SIMPLE iterations)
#   writeControl    timeStep;
#   writeInterval   100000;        (> endTime, so fields are written only at endTime -- minimal I/O)
#
# These four controlDict entries are applied AFTER param-study-common.sh has pinned its STEPS window,
# overriding it for this run; the exit trap in param-study-common.sh restores the original
# controlDict (from system/controlDict.studybak) when the script finishes or is interrupted.
#
# ISOLATED RUN DIRECTORY: before running, the FULL case setup is copied to a fresh, timestamped
# folder occDrivaerRun<STAMP> (beside the case dir, or under $RUN_ROOT) and the production solve runs
# THERE -- so each run's configs, log and probe output are preserved together and the original case
# stays clean. To avoid duplicating the multi-GB mesh, the heavy immutable data is SYMLINKED, not
# copied: constant/polyMesh and the decomposed processor*/ dirs are links back to the original; only
# system/, the initial-field dirs and constant/'s physical-property dicts are copied. The solver
# writes new time dirs back through the processor symlinks into the original decomposition (the
# decomposed field data is shared, not duplicated). build_run_dir does this assembly.
#
# Like param-study-best.sh: the log filename carries a start timestamp and an OPTIONAL user KEYWORD
# (first positional arg) as a suffix, so each production run writes a fresh, preserved, never-skipped,
# taggable log -- e.g.
#   ./param-study-production.sh face-based
#     -> <RUN_ROOT>/occDrivaerRun20260627-143015-face-based/
#          pMG-localized-production-cache-rebuild100-face-based-20260627-143015.log
#        (also symlinked under paramStudyResults/production/ so print_summary/report_cache_reuse find it)
# The keyword is sanitised to [A-Za-z0-9._-]. The "-cache-rebuild" infix keeps the run in the
# preconditioner-cache-reuse report.
#
# NOTE: reset_to_t0 runs in the case dir on the REAL processor*/ dirs (find does not descend the
# symlinks), dropping every written time directory except 0, so this starts a FRESH full run from
# time 0 (re-running overwrites the previous result's time dirs; the timestamped logs/run dirs are
# always preserved).
#
# PROBES / MONITORING: the production run also enables the SAME monitoring function objects the native
# runs use -- the point probes (probes/probes_6xx) plus forceCoeffs, fieldMinMax, solverInfo and
# wallShearStress -- by appending the functions{} block from system/controlDict.noWrite to the run's
# controlDict (the active neoSimpleFoam controlDict has none of its own). Each FO carries its own
# libs(), so no top-level libs change is needed. The block is reverted together with the rest of
# controlDict by param-study-common.sh's exit trap. The probes/probes_* include files referenced by
# that block must exist under system/probes/ on the run host. Set PROD_PROBES=0 to skip the probes.
#
# Tunables (env overrides):
#   BEST_CFG=p-multigrid-localized.json   p-solver configFile basename under system/gko/
#   BEST_REBUILD=100                       preconditionerRebuildInterval (small end safest for the
#                                          staleness-sensitive localized preconditioner)
#   BEST_CACHE=true                        set false to drop the cache keys (regenerate every solve)
#   PROD_END_TIME=4000                     production endTime (SIMPLE iterations at deltaT=1)
#   PROD_WRITE_CONTROL=timeStep            writeControl
#   PROD_WRITE_INTERVAL=100000             writeInterval
#   PROD_PROBES=1                          append controlDict.noWrite's functions{} (probes etc.);
#                                          set 0 to skip
#   RUN_ROOT=<parent of case dir>          where occDrivaerRun<STAMP> is created
#
# Usage:   ./param-study-production.sh
#          ./param-study-production.sh face-based
#          PROD_END_TIME=8000 ./param-study-production.sh long-run

STUDY_TYPE="${STUDY_TYPE:-production}"
source "$(dirname "$0")/param-study-mg-common.sh"

BEST_CFG="${BEST_CFG:-p-multigrid-localized.json}"
BEST_REBUILD="${BEST_REBUILD:-100}"
BEST_CACHE="${BEST_CACHE:-true}"
PROD_END_TIME="${PROD_END_TIME:-4000}"
PROD_WRITE_CONTROL="${PROD_WRITE_CONTROL:-timeStep}"
PROD_WRITE_INTERVAL="${PROD_WRITE_INTERVAL:-100000}"
PROD_PROBES="${PROD_PROBES:-1}"
STAMP="$(date +%Y%m%d-%H%M%S)"

# Optional user keyword (first positional arg) appended to the log name as a suffix, sanitised to
# [A-Za-z0-9._-] (spaces/other chars -> '-') so it is always a safe filename fragment.
KEYWORD_RAW="$1"
KEYWORD="$(printf '%s' "$KEYWORD_RAW" | tr -c 'A-Za-z0-9._-' '-' | sed -E 's/-+/-/g; s/^-|-$//g')"
SUFFIX=""; [ -n "$KEYWORD" ] && SUFFIX="-${KEYWORD}"

# Override param-study-common.sh's short STEPS window with the production run window. startFrom/
# startTime stay at the common.sh values (startTime, 0) so the run marches the full 0 -> endTime.
foamDictionary -entry stopAt        -set endTime               -disableFunctionEntries system/controlDict >/dev/null
foamDictionary -entry endTime       -set "$PROD_END_TIME"      -disableFunctionEntries system/controlDict >/dev/null
foamDictionary -entry writeControl  -set "$PROD_WRITE_CONTROL" -disableFunctionEntries system/controlDict >/dev/null
foamDictionary -entry writeInterval -set "$PROD_WRITE_INTERVAL" -disableFunctionEntries system/controlDict >/dev/null

# Root under which each isolated run directory is created. Default: the parent of the case dir, so
# the run folder sits beside occDrivAerStaticMesh. Override with RUN_ROOT.
RUN_ROOT="${RUN_ROOT:-$(dirname "$PWD")}"

build_run_dir() {
    # Assemble an isolated run directory $1 from the current case: COPY the lightweight, per-run
    # config (system/, the initial-field dirs, constant/ physical-property dicts) and SYMLINK the
    # heavy immutable mesh -- constant/polyMesh and the decomposed processor*/ dirs -- so no multi-GB
    # copy happens. The solver writes new time dirs back through the processor symlinks into the
    # ORIGINAL decomposition (accepted: the decomposed field data is shared, not duplicated; the
    # run's configs / log / probe output are isolated in $1). cwd is the case dir.
    local rd="$1" e p
    mkdir -p "$rd"
    cp -r system "$rd/"
    [ -d 0 ]      && cp -r 0      "$rd/"
    [ -d 0.orig ] && cp -r 0.orig "$rd/"
    rm -f "$rd"/system/*.studybak 2>/dev/null   # drop the harness's dict backups from the copy
    mkdir -p "$rd/constant"
    for e in constant/*; do
        [ -e "$e" ] || continue
        if [ "$(basename "$e")" = polyMesh ]; then
            ln -sfn "$(cd "$e" && pwd)" "$rd/constant/polyMesh"   # symlink the multi-GB mesh
        else
            cp -r "$e" "$rd/constant/"
        fi
    done
    rm -f "$rd"/constant/*.studybak 2>/dev/null
    for p in processor*; do
        [ -d "$p" ] || continue
        ln -sfn "$(cd "$p" && pwd)" "$rd/$p"   # symlink the whole decomposed processor dir
    done
}

add_probes() {
    # Append controlDict.noWrite's functions{} block (probes_6xx + forceCoeffs + fieldMinMax +
    # solverInfo + wallShearStress) to the controlDict $1 -- "probes like in controlDict.noWrite".
    # Each FO carries its own libs(), so no top-level libs change is needed. Sourced verbatim so it
    # stays in sync with controlDict.noWrite.
    local cd="$1" src="system/controlDict.noWrite"
    if [ "$PROD_PROBES" != "1" ]; then
        echo "   (PROD_PROBES=$PROD_PROBES -> not adding probes/monitoring functions)"; return
    fi
    if grep -qE '^[[:space:]]*functions' "$cd"; then
        echo "   (controlDict already has a functions{} block -- leaving it as-is)"; return
    fi
    if [ ! -f "$src" ]; then
        echo "   !! $src not found -- production run will record no probes"; return
    fi
    { echo; sed -n '/^functions/,/^}/p' "$src"; } >> "$cd"
    echo "   (added monitoring functions{} from $src to the production controlDict)"
    [ -d system/probes ] \
        || echo "   !! system/probes/ not found -- the probes/probes_* includes will fail at run time; provide the probe files on the run host"
}

ensure_mg_variants

run_production() {
    if [ ! -f "system/gko/$BEST_CFG" ]; then
        echo "!! no best-practice config 'system/gko/$BEST_CFG' -- skipping"; return
    fi
    local tag; if [ "$BEST_CACHE" = "true" ]; then tag="cache-rebuild${BEST_REBUILD}"; else tag="nocache"; fi
    # The -${STAMP} timestamp is appended to the log name below (trailing), matching the
    # paramStudyResults/<type>/<run-name><timestamp>.log convention; $name stays stamp-free
    # so it is a stable key for RUNS / RUN_LOG / the summary.
    local name="pMG-localized-production-${tag}${SUFFIX}"

    # 1. Assemble the isolated run directory: copy the configs, symlink the mesh + processor dirs.
    local RUN_DIR="$RUN_ROOT/occDrivaerRun${STAMP}${SUFFIX}"
    echo "================================================================"
    echo " building run dir: $RUN_DIR"
    build_run_dir "$RUN_DIR"

    # 2. Install the chosen p-solver fvSolution into the run dir's system/ (configFile + cache keys),
    #    exactly like the cache study: swap the configFile path and append the cache keys, leaving the
    #    template's l1ScaledResidual/tolerance/relTol intact. sed on the configFile line -- NOT
    #    foamDictionary, which re-tokenises the "system/gko/..." path into separate words.
    if [ "$BEST_CACHE" = "true" ]; then
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_CFG;\n        cacheSolver      true;\n        preconditionerRebuildInterval $BEST_REBUILD;#" \
            "$MG_TEMPLATE" > "$RUN_DIR/system/fvSolution"
    else
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_CFG;#" "$MG_TEMPLATE" > "$RUN_DIR/system/fvSolution"
    fi

    # 3. Probes/monitoring into the run dir's controlDict (the copy already carries the production
    #    window from the foamDictionary overrides above).
    add_probes "$RUN_DIR/system/controlDict"

    # 4. Run in the run dir. A SUBSHELL keeps the MAIN shell in the case dir so param-study-common.sh's
    #    restore trap (relative paths) still works. reset_to_t0 runs here in the main shell on the REAL
    #    processor*/ dirs (find does not descend the symlinks), clearing old time dirs; the subshell
    #    then writes fresh ones back through the processor symlinks. The log lives in the run dir; a
    #    symlink under $RESULTS makes print_summary / report_cache_reuse find it unchanged.
    # name already carries ${STAMP}, so the real log + its $RESULTS symlink both follow the
    # paramStudyResults/<type>/<run-name><timestamp>.log convention. Register the log so
    # print_summary / report_cache_reuse (which read RUN_LOG) find it.
    local log="$RUN_DIR/${name}-${STAMP}.log"
    : > "$log"
    ln -sf "$log" "$RESULTS/${name}-${STAMP}.log"
    RUN_LOG["$name"]="$log"
    echo "================================================================"
    echo " $name : PRODUCTION p = Ginkgo LOCALIZED MG ($BEST_CFG), cache=${BEST_CACHE}, rebuildInterval=${BEST_REBUILD}, endTime=${PROD_END_TIME}${KEYWORD:+, keyword=${KEYWORD}}"
    echo "   run dir -> $RUN_DIR"
    echo "   log     -> $log"
    echo "================================================================"
    reset_to_t0
    local t0=$SECONDS
    # kokkos_launch forwards KOKKOS_TOOLS_LIBS to the ranks when KOKKOS_TOOL is set (profiler output
    # -- .dat files / stdout summaries -- lands in $RUN_DIR, the run cwd).
    ( cd "$RUN_DIR" && kokkos_launch "$BIN" ) > "$log" 2>&1 &
    local pid=$!
    tail_window "$pid" "$log"
    wait "$pid"; local rc=$?
    # Gather kokkos-tools output next to the run log in $RUN_DIR (the space-time-stack report -> a
    # <stem>.kokkos-profile.txt sidecar; any .dat/json/perfetto files moved to <stem>.<file>). RUN_DIR
    # is freshly built, so no marker is needed to scope the file move.
    collect_kokkos_output "$log" "$RUN_DIR"
    echo "   exit=$rc  wall=$((SECONDS - t0))s  steps=$(grep -c '^Time = ' "$log")"
    RUNS+=("$name")
}

run_production

print_summary
report_cache_reuse
