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
# Like param-study-best.sh: the log filename carries a start timestamp and an OPTIONAL user KEYWORD
# (first positional arg) as a suffix, so each production run writes a fresh, preserved, never-skipped,
# taggable log -- e.g.
#   ./param-study-production.sh face-based
#     -> paramStudy/results/pMG-localized-production-cache-rebuild100-20260627-143015-face-based.log
# The keyword is sanitised to [A-Za-z0-9._-]. The "-cache-rebuild" infix keeps the run in the
# preconditioner-cache-reuse report.
#
# NOTE: run_one calls reset_to_t0, which drops every written time directory except 0 before the run,
# so this starts a FRESH full run from time 0 (re-running overwrites the previous result's time dirs;
# the timestamped logs are always preserved). Set FORCE=1 only matters for log skipping, which the
# timestamp already defeats.
#
# Tunables (env overrides):
#   BEST_CFG=p-multigrid-localized.json   p-solver configFile basename under system/gko/
#   BEST_REBUILD=100                       preconditionerRebuildInterval (small end safest for the
#                                          staleness-sensitive localized preconditioner)
#   BEST_CACHE=true                        set false to drop the cache keys (regenerate every solve)
#   PROD_END_TIME=4000                     production endTime (SIMPLE iterations at deltaT=1)
#   PROD_WRITE_CONTROL=timeStep            writeControl
#   PROD_WRITE_INTERVAL=100000             writeInterval
#
# Usage:   ./param-study-production.sh
#          ./param-study-production.sh face-based
#          PROD_END_TIME=8000 ./param-study-production.sh long-run

source "$(dirname "$0")/param-study-mg-common.sh"

BEST_CFG="${BEST_CFG:-p-multigrid-localized.json}"
BEST_REBUILD="${BEST_REBUILD:-100}"
BEST_CACHE="${BEST_CACHE:-true}"
PROD_END_TIME="${PROD_END_TIME:-4000}"
PROD_WRITE_CONTROL="${PROD_WRITE_CONTROL:-timeStep}"
PROD_WRITE_INTERVAL="${PROD_WRITE_INTERVAL:-100000}"
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

ensure_mg_variants

run_production() {
    # Build the p-block exactly like the cache study's run_cache_interval: swap the configFile path
    # and (when caching) append the cache keys, leaving the template's l1ScaledResidual/tolerance/
    # relTol intact (the working fp64 stop). sed on the configFile line -- NOT foamDictionary, which
    # re-tokenises the "system/gko/..." path into separate words.
    if [ ! -f "system/gko/$BEST_CFG" ]; then
        echo "!! no best-practice config 'system/gko/$BEST_CFG' -- skipping"; return
    fi
    local tag; if [ "$BEST_CACHE" = "true" ]; then tag="cache-rebuild${BEST_REBUILD}"; else tag="nocache"; fi
    local name="pMG-localized-production-${tag}-${STAMP}${SUFFIX}"
    if [ "$BEST_CACHE" = "true" ]; then
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_CFG;\n        cacheSolver      true;\n        preconditionerRebuildInterval $BEST_REBUILD;#" \
            "$MG_TEMPLATE" > "$TMP_FVSOL"
    else
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_CFG;#" "$MG_TEMPLATE" > "$TMP_FVSOL"
    fi
    run_one "$name" "$TMP_FVSOL" \
        "PRODUCTION: p = Ginkgo LOCALIZED MG ($BEST_CFG), cache=${BEST_CACHE}, rebuildInterval=${BEST_REBUILD}, endTime=${PROD_END_TIME}, started ${STAMP}${KEYWORD:+, keyword=${KEYWORD}}"
}

run_production

print_summary
report_cache_reuse
