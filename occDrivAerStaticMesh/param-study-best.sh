#!/bin/bash
#
# BEST-PRACTICE run(s) for occDrivAreStaticMesh.
#
# Runs TWO configurations back to back over the same window:
#   1. the best-practice pressure solver (pMG-localized-cache-rebuild100, below); and
#   2. a reference Ginkgo PCG (solver::Cg) + DIAGONAL (Jacobi) preconditioner run --
#      system/gko/p-cg.json (Schwarz{Jacobi} for the distributed solve), no MG, no solver cache --
#      so the best-practice multigrid run has a plain-PCG baseline to compare against. Same case3
#      U/k/omega block and l1ScaledResidual fp64 stop; only the p-solver configFile differs.
#
# The best-practice configuration is the pMG-localized-cache-rebuild100 setup from
# param-study-mg.sh, which is the best-practice pressure solver established by the sweeps:
#
#   p = Ginkgo Cg + LOCALIZED Multigrid preconditioner (Schwarz{Multigrid(local)}, max_levels=10,
#       local-Jacobi smoother)  --  system/gko/p-multigrid-localized.json
#   solver cache ON: cacheSolver=true, preconditionerRebuildInterval=100 (the generated solver +
#       its MG hierarchy are cached and refreshed in place via update_matrix_value, rebuilt from
#       scratch only every 100th solve)
#   outer stop: NeoN l1ScaledResidual (tolerance 1e-7, relTol 0.01) -- the fp64 path.
#   U/k/omega: PBiCGStab/diagonal smoothSolver (case3 template).
#
# This is the fp64 (double) configuration ON PURPOSE: the mixed-precision variants (innerPrecision
# float/bfloat16 and the per-node value_type configs) ABORT at solver setup for the localized
# distributed-Schwarz construction -- float throws gko::NotSupported in Schwarz::extract_local_matrix
# (gko::as can't cast the fp64 distributed matrix), bfloat16 isn't even in distributed Schwarz's
# config value_type_list_base. See param-study-mp.sh for that investigation. So "best practice" =
# the localized cached fp64 run.
#
# TIMESTAMP + KEYWORD: the log filename carries a start timestamp (date +%Y%m%d-%H%M%S) and an
# optional user KEYWORD given as the first positional argument, appended as a suffix -- so every
# invocation writes a fresh, preserved, never-skipped log that you can tag, e.g.
#   ./param-study-best.sh face-based
#     -> paramStudyResults/best-practice/pMG-localized-best-cache-rebuild100-face-based-20260627-143015.log
# The keyword is sanitised to [A-Za-z0-9._-] (other chars -> '-'). The "-cache-rebuild" infix keeps
# the run in the preconditioner-cache-reuse report.
#
# Tunables (env overrides):
#   BEST_CFG=p-multigrid-localized.json   the p-solver configFile basename under system/gko/
#   BEST_REBUILD=100                       preconditionerRebuildInterval (small end is safest for
#                                          the staleness-sensitive localized preconditioner)
#   BEST_CACHE=true                        set false to drop the cache keys (regenerate every solve)
#   BEST_PCG_CFG=p-cg.json                 the PCG/diagonal reference configFile under system/gko/
#   KOKKOS_TOOL=<name>                     load a kokkos-tools profiler connector for the run (off by
#                                          default). Needs the binary built with
#                                          NEOFOAM_ENABLE_KOKKOS_TOOLS=ON (the profiling builds have
#                                          it). Names: simple-kernel-timer, space-time-stack,
#                                          memory-high-water-mark, memory-usage, memory-events,
#                                          chrome-tracing, perfetto-connector, kernel-logger.
#                                          space-time-stack / memory-* summarise into the run log;
#                                          simple-kernel-timer writes <host>-<pid>.dat per rank to the
#                                          run cwd (read with kp_reader next to the connector .so).
#
# Usage:   ./param-study-best.sh                 # log suffixed with timestamp only
#          ./param-study-best.sh face-based       # ...timestamp + "-face-based" keyword suffix
#          BEST_REBUILD=20 ./param-study-best.sh tuned
#          BEST_CACHE=false ./param-study-best.sh nocache-baseline
#          KOKKOS_TOOL=space-time-stack ./param-study-best.sh profiled   # kokkos-tools profiler

STUDY_TYPE="${STUDY_TYPE:-best-practice}"
source "$(dirname "$0")/param-study-mg-common.sh"

BEST_CFG="${BEST_CFG:-p-multigrid-localized.json}"
BEST_REBUILD="${BEST_REBUILD:-100}"
BEST_CACHE="${BEST_CACHE:-true}"
BEST_PCG_CFG="${BEST_PCG_CFG:-p-cg.json}"
STAMP="$(date +%Y%m%d-%H%M%S)"
# run_one appends the -<timestamp> to the log name now, so the run names below drop their
# embedded STAMP. Force-fresh by default so each invocation still writes a preserved,
# never-skipped log (the timestamp keeps successive runs distinct).
FORCE="${FORCE:-1}"

# Optional user keyword (first positional arg) appended to the log name as a suffix, e.g.
#   ./param-study-best.sh face-based  ->  pMG-localized-best-cache-rebuild100-<stamp>-face-based.log
# Sanitised to [A-Za-z0-9._-] (spaces/other chars -> '-') so it is always a safe filename fragment.
KEYWORD_RAW="$1"
KEYWORD="$(printf '%s' "$KEYWORD_RAW" | tr -c 'A-Za-z0-9._-' '-' | sed -E 's/-+/-/g; s/^-|-$//g')"
SUFFIX=""; [ -n "$KEYWORD" ] && SUFFIX="-${KEYWORD}"

ensure_mg_variants

run_best() {
    # Build the p-block exactly like the cache study's run_cache_interval: swap the configFile path
    # and (when caching) append the cache keys, leaving the template's l1ScaledResidual/tolerance/
    # relTol intact (the working fp64 stop). sed on the configFile line -- NOT foamDictionary, which
    # re-tokenises the "system/gko/..." path into separate words.
    if [ ! -f "system/gko/$BEST_CFG" ]; then
        echo "!! no best-practice config 'system/gko/$BEST_CFG' -- skipping"; return
    fi
    local tag; if [ "$BEST_CACHE" = "true" ]; then tag="cache-rebuild${BEST_REBUILD}"; else tag="nocache"; fi
    local name="pMG-localized-best-${tag}${SUFFIX}"
    if [ "$BEST_CACHE" = "true" ]; then
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_CFG;\n        cacheSolver      true;\n        preconditionerRebuildInterval $BEST_REBUILD;#" \
            "$MG_TEMPLATE" > "$TMP_FVSOL"
    else
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_CFG;#" "$MG_TEMPLATE" > "$TMP_FVSOL"
    fi
    run_one "$name" "$TMP_FVSOL" \
        "BEST-PRACTICE: p = Ginkgo LOCALIZED MG ($BEST_CFG), cache=${BEST_CACHE}, rebuildInterval=${BEST_REBUILD}, started ${STAMP}${KEYWORD:+, keyword=${KEYWORD}}"
}

run_pcg_diagonal() {
    # Second run: a plain Ginkgo PCG (solver::Cg) with a DIAGONAL (Jacobi, max_block_size=1)
    # preconditioner -- system/gko/p-cg.json, which wraps the Jacobi in a Schwarz for the distributed
    # solve. No MG and no solver cache: a Jacobi-preconditioned Cg has no expensive hierarchy to reuse,
    # so this is a plain configFile swap (no cacheSolver/rebuild keys). The name omits the
    # "-cache-rebuild" infix, so report_cache_reuse correctly ignores it. Same case3 U/k/omega block
    # and l1ScaledResidual fp64 stop as run_best; only the p-solver configFile differs.
    if [ ! -f "system/gko/$BEST_PCG_CFG" ]; then
        echo "!! no PCG/diagonal config 'system/gko/$BEST_PCG_CFG' -- skipping"; return
    fi
    local name="pPCG-diagonal-best${SUFFIX}"
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_PCG_CFG;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL"
    run_one "$name" "$TMP_FVSOL" \
        "BEST-PRACTICE baseline: p = Ginkgo PCG (Cg) + diagonal (Jacobi) preconditioner ($BEST_PCG_CFG), started ${STAMP}${KEYWORD:+, keyword=${KEYWORD}}"
}

run_best
run_pcg_diagonal

print_summary
report_cache_reuse
