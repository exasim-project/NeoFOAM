#!/bin/bash
#
# PMIS coarsening parameter study for occDrivAreStaticMesh.
#
# Sweeps the LOCALIZED Multigrid pressure preconditioner with PMIS coarsening (multigrid::Pmis,
# Ginkgo PR #2037) over two axes:
#
#   max_levels         {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}   (MG depth)
#   strength_threshold {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}
#                                                              (PMIS strength-of-connection cut)
#
# Each run is the localized PMIS config (system/gko/p-multigrid-localized-pmis.json, i.e. the
# best-practice localized Schwarz{Multigrid(local)} with the inner mg_level switched Pgm -> PMIS),
# with skip_sorting=true on the PMIS node and SOLVER CACHE ON (cacheSolver=true,
# preconditionerRebuildInterval=REBUILD, default 100). Caching matters here: PMIS coarsening runs on
# the host ReferenceExecutor and is copied to the device (the device classify kernels are incomplete
# upstream -- see param-study-mg.sh `pmis` / ginkgo_pmis_enable.patch), so the cache amortizes that
# host detour over REBUILD solves. Only the U/k/omega block + l1 stop are shared (case3 template);
# only the pressure configFile + its max_levels/strength_threshold differ across runs.
#
# REQUIRES the PMIS-enabled Ginkgo (ginkgo_pmis_enable.patch) which lives only in the profiling
# build, so NEON_BUILD defaults to `profiling` here (overridable).
#
# The per-(L,threshold) configs are generated into system/gko/ as
#   p-multigrid-localized-pmis-L<L>-th<TTT>.json   (TTT = threshold with '.'->'p', e.g. 0p25)
# from the localized PMIS base, and overwritten on each run.
#
# Modes (positional arg):
#   (none) | grid   full LEVELS x THRESHOLDS grid (default: 10 x 9 = 90 runs -- LONG)
#   levels          sweep LEVELS only, at fixed strength_threshold=PMIS_THRESHOLD_FIXED (0.25)
#   thresholds      sweep THRESHOLDS only, at fixed max_levels=PMIS_LEVEL_FIXED (10)
#
# Tunables (env overrides):
#   PMIS_LEVELS="2 4 6 8 10 12 14 16 18 20"                    max_levels sweep values
#   PMIS_THRESHOLDS="0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5"  strength_threshold sweep values
#   PMIS_LEVEL_FIXED=10        fixed max_levels for the `thresholds` mode
#   PMIS_THRESHOLD_FIXED=0.25  fixed strength_threshold for the `levels` mode
#   PMIS_SKIP_SORTING=true     PMIS skip_sorting (assumes the local block is already sorted)
#   REBUILD=100                preconditionerRebuildInterval (solver cache)
#   NEON_BUILD=profiling       build to run (PMIS only exists in the profiling build)
#   KOKKOS_TOOL=<name>         load a kokkos-tools profiler for each run
#
# Usage:   ./param-study-mg-pmis.sh                 # full 10x9 grid (90 runs)
#          ./param-study-mg-pmis.sh levels          # 10 runs, threshold fixed 0.25
#          ./param-study-mg-pmis.sh thresholds      # 9 runs, max_levels fixed 10
#          PMIS_LEVELS="10" ./param-study-mg-pmis.sh thresholds   # threshold sweep at L10
#          REBUILD=50 ./param-study-mg-pmis.sh levels

# PMIS lives only in the profiling Ginkgo build; pin NEON_BUILD before common.sh resolves BIN.
export NEON_BUILD="${NEON_BUILD:-profiling}"

STUDY_TYPE="${STUDY_TYPE:-pmis-sweep}"
source "$(dirname "$0")/param-study-mg-common.sh"

ensure_mg_variants   # ensures the localized PMIS base config exists

BASE_CFG="p-multigrid-localized-pmis.json"
REBUILD="${REBUILD:-100}"
PMIS_SKIP_SORTING="${PMIS_SKIP_SORTING:-true}"
PMIS_LEVELS=(${PMIS_LEVELS:-2 4 6 8 10 12 14 16 18 20})
PMIS_THRESHOLDS=(${PMIS_THRESHOLDS:-0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5})
PMIS_LEVEL_FIXED="${PMIS_LEVEL_FIXED:-10}"
PMIS_THRESHOLD_FIXED="${PMIS_THRESHOLD_FIXED:-0.25}"

if [ ! -f "system/gko/$BASE_CFG" ]; then
    echo "!! base PMIS config 'system/gko/$BASE_CFG' not found -- regenerate via gen-mg-variants.py"; exit 1
fi

# Write a (max_levels, strength_threshold) variant of the localized PMIS base: deep-copy the base
# and set preconditioner.local_solver.max_levels + the inner PMIS mg_level (strength_threshold,
# skip_sorting). $1=levels $2=threshold $3=skip_sorting(true|false) $4=output basename in system/gko/.
gen_pmis_cfg() {
    local lev="$1" thr="$2" skip="$3" out="$4"
    python3 - "system/gko/$BASE_CFG" "$lev" "$thr" "$skip" "system/gko/$out" <<'PY'
import json, sys
base, lev, thr, skip, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
cfg = json.load(open(base))
try:
    mg = cfg["preconditioner"]["local_solver"]
except (KeyError, TypeError):
    sys.exit("!! base config has no preconditioner.local_solver (not a localized config)")
mg["max_levels"] = lev
mg["mg_level"] = [{
    "type": "multigrid::Pmis",
    "strength_threshold": thr,
    "skip_sorting": skip.lower() == "true",
}]
with open(out, "w") as f:
    json.dump(cfg, f, indent=4)
    f.write("\n")
PY
}

# Run one (level, threshold) point: generate its config, then run_cache_interval (cacheSolver=true +
# preconditionerRebuildInterval=REBUILD), naming the run pMG-pmis-L<L>-th<TTT>-cache-rebuild<REBUILD>.
run_pmis_point() {
    local lev="$1" thr="$2"
    local tlabel="${thr//./p}"                       # 0.25 -> 0p25 (safe run-name fragment)
    local out="p-multigrid-localized-pmis-L${lev}-th${tlabel}.json"
    if ! gen_pmis_cfg "$lev" "$thr" "$PMIS_SKIP_SORTING" "$out"; then
        echo "!! failed to generate $out -- skipping"; return
    fi
    run_cache_interval "pmis-L${lev}-th${tlabel}" "$out" "$REBUILD"
}

MODE="${1:-grid}"

echo "################################################################"
echo " PMIS coarsening sweep ($MODE) on $BASE_CFG  [skip_sorting=$PMIS_SKIP_SORTING, rebuild=$REBUILD]"
case "$MODE" in
    levels)
        echo "   max_levels: ${PMIS_LEVELS[*]}   (threshold fixed ${PMIS_THRESHOLD_FIXED})"
        echo "################################################################"
        for lev in "${PMIS_LEVELS[@]}"; do
            run_pmis_point "$lev" "$PMIS_THRESHOLD_FIXED"
        done
        ;;
    thresholds)
        echo "   strength_threshold: ${PMIS_THRESHOLDS[*]}   (max_levels fixed ${PMIS_LEVEL_FIXED})"
        echo "################################################################"
        for thr in "${PMIS_THRESHOLDS[@]}"; do
            run_pmis_point "$PMIS_LEVEL_FIXED" "$thr"
        done
        ;;
    grid|"")
        echo "   GRID: levels {${PMIS_LEVELS[*]}} x thresholds {${PMIS_THRESHOLDS[*]}}"
        echo "   = $(( ${#PMIS_LEVELS[@]} * ${#PMIS_THRESHOLDS[@]} )) runs (LONG)"
        echo "################################################################"
        for lev in "${PMIS_LEVELS[@]}"; do
            for thr in "${PMIS_THRESHOLDS[@]}"; do
                run_pmis_point "$lev" "$thr"
            done
        done
        ;;
    *)
        echo "!! unknown mode '$MODE' (use: grid | levels | thresholds)"; exit 2
        ;;
esac

print_summary
report_cache_reuse
