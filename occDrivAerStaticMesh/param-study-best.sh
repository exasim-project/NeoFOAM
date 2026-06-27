#!/bin/bash
#
# BEST-PRACTICE single run for occDrivAreStaticMesh.
#
# Runs exactly ONE selected configuration -- the pMG-localized-cache-rebuild100 setup from
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
#     -> paramStudy/results/pMG-localized-best-cache-rebuild100-20260627-143015-face-based.log
# The keyword is sanitised to [A-Za-z0-9._-] (other chars -> '-'). The "-cache-rebuild" infix keeps
# the run in the preconditioner-cache-reuse report.
#
# Tunables (env overrides):
#   BEST_CFG=p-multigrid-localized.json   the p-solver configFile basename under system/gko/
#   BEST_REBUILD=100                       preconditionerRebuildInterval (small end is safest for
#                                          the staleness-sensitive localized preconditioner)
#   BEST_CACHE=true                        set false to drop the cache keys (regenerate every solve)
#
# Usage:   ./param-study-best.sh                 # log suffixed with timestamp only
#          ./param-study-best.sh face-based       # ...timestamp + "-face-based" keyword suffix
#          BEST_REBUILD=20 ./param-study-best.sh tuned
#          BEST_CACHE=false ./param-study-best.sh nocache-baseline

source "$(dirname "$0")/param-study-mg-common.sh"

BEST_CFG="${BEST_CFG:-p-multigrid-localized.json}"
BEST_REBUILD="${BEST_REBUILD:-100}"
BEST_CACHE="${BEST_CACHE:-true}"
STAMP="$(date +%Y%m%d-%H%M%S)"

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
    local name="pMG-localized-best-${tag}-${STAMP}${SUFFIX}"
    if [ "$BEST_CACHE" = "true" ]; then
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_CFG;\n        cacheSolver      true;\n        preconditionerRebuildInterval $BEST_REBUILD;#" \
            "$MG_TEMPLATE" > "$TMP_FVSOL"
    else
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_CFG;#" "$MG_TEMPLATE" > "$TMP_FVSOL"
    fi
    run_one "$name" "$TMP_FVSOL" \
        "BEST-PRACTICE: p = Ginkgo LOCALIZED MG ($BEST_CFG), cache=${BEST_CACHE}, rebuildInterval=${BEST_REBUILD}, started ${STAMP}${KEYWORD:+, keyword=${KEYWORD}}"
}

run_best

print_summary
report_cache_reuse
