#!/bin/bash
#
# MIXED-PRECISION PRODUCTION comparison for occDrivAreStaticMesh.
#
# A copy of param-study-production.sh that runs the SAME localized cached best-practice solver
# (Ginkgo Cg + LOCALIZED Multigrid preconditioner, max_levels=10, outer Cg + l1ScaledResidual stop
# fp64, solver cache ON, validated CG-10 coarse solve) over a PROD_END_TIME window (default 1000
# SIMPLE iterations), back to back for TWO preconditioner precisions:
#
#   double : fp64 MG preconditioner  (system/gko/p-multigrid-localized.json)
#   float  : float32 MG preconditioner (system/gko/p-multigrid-localized-precfloat.json) -- the
#            preconditioner runs in float (outer Cg + stop stay fp64); needs the float distributed-
#            Schwarz fix (ginkgo_schwarz_update_matrix_value.patch).
#
# Only the preconditioner value_type differs between the two runs, so this is a like-for-like
# float-vs-fp64 production comparison (wall time, p-iterations, forces). Each variant gets its own
# isolated, timestamped run directory and a distinct run name (the "-double"/"-float" infix), so the
# two never collide and both are preserved.
#
# Everything else mirrors param-study-production.sh:
#   * each run is assembled into a fresh occDrivaerRun<STAMP>-<label> dir (configs + initial fields
#     COPIED, the multi-GB mesh + decomposed processor*/ dirs SYMLINKED -- no mesh duplication);
#   * the production window (stopAt endTime, endTime=PROD_END_TIME, writeInterval > endTime so fields
#     are written only at endTime) overrides the harness STEPS pin and is reverted by the exit trap;
#   * aerodynamic FORCES are recorded via the GPU-native neoForceCoeffs function object
#     (system/neoForceCoeffs) into each run dir's postProcessing/neoForceCoeffs/coefficient.dat -- the
#     only force path that works with neoSimpleFoam (the OF-native FOs abort on NO_REGISTER fields);
#   * reset_to_t0 runs between the two variants on the REAL processor*/ dirs, so each marches a FRESH
#     0 -> endTime; the per-variant forces live in their own run dirs and are preserved.
#
# Run dir / log example (KEYWORD optional, first positional arg, sanitised to [A-Za-z0-9._-]):
#   ./param-study-production-mp.sh tag
#     -> <RUN_ROOT>/occDrivaerRun<STAMP>-tag-double/
#          pMG-localized-production-double-cache-rebuild100-tag-<STAMP>.log
#     -> <RUN_ROOT>/occDrivaerRun<STAMP>-tag-float/
#          pMG-localized-production-float-cache-rebuild100-tag-<STAMP>.log
#   (both symlinked under paramStudyResults/production-mp/ so print_summary/report_cache_reuse find them).
# The "-cache-rebuild" infix keeps each run in the preconditioner-cache-reuse report.
#
# Tunables (env overrides):
#   MP_VARIANTS="double:p-multigrid-localized.json float:p-multigrid-localized-precfloat.json"
#                                          the precision sweep, as "label:configFile.json" pairs
#   PROD_END_TIME=1000                      production endTime (SIMPLE iterations at deltaT=1)
#   BEST_COARSE_SOLVER=cg                    MG coarsest_solver: cg (validated best practice) |
#                                          jacobi (damped-Ir sweeps) | off (leave the config's own)
#   BEST_COARSE_ITERS=10                    coarse-solver iteration count (10 = the sweet spot)
#   BEST_REBUILD=100                        preconditionerRebuildInterval
#   BEST_CACHE=true                         set false to drop the cache keys (regenerate every solve)
#   PROD_WRITE_CONTROL=timeStep             writeControl
#   PROD_WRITE_INTERVAL=100000              writeInterval (> endTime -> write only at endTime)
#   PROD_PROBES=1                           append the GPU-native neoForceCoeffs functions{} block
#   RUN_ROOT=<parent of case dir>           where occDrivaerRun<STAMP>-<label> is created
#
# Usage:   ./param-study-production-mp.sh                 # double + float, 1000 steps
#          ./param-study-production-mp.sh tuned            # ...with a "-tuned" keyword suffix
#          PROD_END_TIME=2000 ./param-study-production-mp.sh
#          MP_VARIANTS="float:p-multigrid-localized-precfloat.json" ./param-study-production-mp.sh  # float only

STUDY_TYPE="${STUDY_TYPE:-production-mp}"
source "$(dirname "$0")/param-study-mg-common.sh"

BEST_REBUILD="${BEST_REBUILD:-100}"
BEST_CACHE="${BEST_CACHE:-true}"
# Best-practice MG coarsest_solver (coarse-grid-solver sweep, paramStudyResults/coarse-solve): a
# Jacobi-preconditioned CG run for 10 iterations on the local coarse matrix. BEST_COARSE_SOLVER=off
# keeps whatever coarsest_solver gen-mg-variants.py bakes into the config.
BEST_COARSE_SOLVER="${BEST_COARSE_SOLVER:-cg}"
BEST_COARSE_ITERS="${BEST_COARSE_ITERS:-10}"
PROD_END_TIME="${PROD_END_TIME:-1000}"
PROD_WRITE_CONTROL="${PROD_WRITE_CONTROL:-timeStep}"
PROD_WRITE_INTERVAL="${PROD_WRITE_INTERVAL:-100000}"
PROD_PROBES="${PROD_PROBES:-1}"
STAMP="$(date +%Y%m%d-%H%M%S)"

# The precision sweep, as "label:configFile.json" pairs:
#   double         = fp64 Cg + localized MG PRECONDITIONER (best practice)
#   float          = Cg + localized MG PRECONDITIONER in float32 (outer Cg/stop stay fp64)
#   mgsolver-float = NON-LOCALIZED Multigrid as the TOP-LEVEL SOLVER (no outer Cg), entirely in
#                    float32, with a relative ResidualNorm stop (the whole solve is float, so the
#                    fp64 l1ScaledResidual stop cannot be used). CG-10 coarse is baked in below.
MP_VARIANTS=(${MP_VARIANTS:-
    "double:p-multigrid-localized.json"
    "float:p-multigrid-localized-precfloat.json"
    "mgsolver-float:p-multigrid-solver-precfloat.json"})

# Optional user keyword (first positional arg) appended to the log name as a suffix, sanitised to
# [A-Za-z0-9._-] (spaces/other chars -> '-') so it is always a safe filename fragment.
KEYWORD_RAW="$1"
KEYWORD="$(printf '%s' "$KEYWORD_RAW" | tr -c 'A-Za-z0-9._-' '-' | sed -E 's/-+/-/g; s/^-|-$//g')"
SUFFIX=""; [ -n "$KEYWORD" ] && SUFFIX="-${KEYWORD}"

# Override param-study-common.sh's short STEPS window with the production run window. startFrom/
# startTime stay at the common.sh values (startTime, 0) so each run marches the full 0 -> endTime.
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
    # Append a functions{} block to the run controlDict $1 that monitors aerodynamic FORCES via the
    # GPU-native neoForceCoeffs function object (system/neoForceCoeffs, libNeoFOAM). neoForceCoeffs
    # reads p/U/nut straight from NeoN's on-device VectorCollection and writes
    # postProcessing/neoForceCoeffs/<t0>/coefficient.dat (Cd/Cl/Cs + axle splits + moments, matching
    # OpenFOAM's forceCoeffs). This is the ONLY force/monitoring path that works with neoSimpleFoam:
    # the OF-native forceCoeffs/forces/probes/... FOs look their fields up in the OF objectRegistry,
    # but neoSimpleFoam keeps the solution in NeoN GPU fields (createFields.H NO_REGISTER), so those
    # abort at startup. neoForces resolves patches by literal name (no regex), so system/neoForceCoeffs
    # lists the wall patches explicitly.
    local cd="$1" src="system/neoForceCoeffs"
    if [ "$PROD_PROBES" != "1" ]; then
        echo "   (PROD_PROBES=$PROD_PROBES -> not adding the neoForceCoeffs monitoring function)"; return
    fi
    if grep -qE '^[[:space:]]*functions' "$cd"; then
        echo "   (controlDict already has a functions{} block -- leaving it as-is)"; return
    fi
    if [ ! -f "$src" ]; then
        echo "   !! $src not found -- production run will record no forces"; return
    fi
    { echo; echo "functions"; echo "{"; echo "    #include \"neoForceCoeffs\""; echo "}"; } >> "$cd"
    echo "   (added GPU-native neoForceCoeffs functions{} to the production controlDict)"
}

ensure_mg_variants

# Replace the MG coarsest_solver (preconditioner.local_solver.coarsest_solver) of a localized config
# with the best-practice coarse solver (BEST_COARSE_SOLVER/BEST_COARSE_ITERS). Writes a sibling
# <src-stem>-coarse-<solver><n>.json under system/gko/ and echoes its basename on success; on any
# failure it warns to stderr and echoes the original basename unchanged. Mirrors the coarse-solver
# generation in param-study-mg-coarse-solve.sh's gen_coarse_cfg (CG + local Jacobi, or damped Ir).
bake_coarse_solver() {
    local src="$1" solver="$BEST_COARSE_SOLVER" n="$BEST_COARSE_ITERS"
    local out="${src%.json}-coarse-${solver}${n}.json"
    if python3 - "system/gko/$src" "$solver" "$n" "system/gko/$out" <<'PY' 2>/dev/null
import json, sys
src, solver, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = json.load(open(src))
if solver == "cg":
    coarse = {
        "type": "solver::Cg",
        "preconditioner": {"type": "preconditioner::Jacobi", "max_block_size": 1},
        "criteria": [{"type": "Iteration", "max_iters": n}],
    }
elif solver == "jacobi":
    coarse = {
        "type": "solver::Ir",
        "relaxation_factor": 0.9,
        "solver": {"type": "preconditioner::Jacobi", "max_block_size": 1},
        "criteria": [{"type": "Iteration", "max_iters": n}],
    }
else:
    sys.exit("unknown coarse solver '%s'" % solver)
# Localized configs keep the MG coarsest_solver under preconditioner.local_solver; the non-localized
# Multigrid-as-solver config has it at the top level. Handle both.
if isinstance(cfg.get("preconditioner", {}).get("local_solver", {}).get("coarsest_solver"), dict):
    cfg["preconditioner"]["local_solver"]["coarsest_solver"] = coarse
elif isinstance(cfg.get("coarsest_solver"), dict):
    cfg["coarsest_solver"] = coarse
else:
    sys.exit("no coarsest_solver to replace in %s" % src)
with open(out, "w") as f:
    json.dump(cfg, f, indent=4); f.write("\n")
PY
    then
        echo "$out"
    else
        echo "!! failed to bake coarse solver into $src -- using it unchanged" >&2
        echo "$src"
    fi
}

run_production() {
    # Uses the per-variant MP_LABEL + BEST_CFG set by the loop below.
    if [ ! -f "system/gko/$BEST_CFG" ]; then
        echo "!! [$MP_LABEL] no production config 'system/gko/$BEST_CFG' -- skipping"; return
    fi
    local tag; if [ "$BEST_CACHE" = "true" ]; then tag="cache-rebuild${BEST_REBUILD}"; else tag="nocache"; fi
    # name carries the precision label ("-double"/"-float") so the two variants never collide and both
    # land in RUNS / RUN_LOG / the cache-reuse report. $name stays stamp-free (stable summary key).
    local name="pMG-localized-production-${MP_LABEL}-${tag}${SUFFIX}"

    # 1. Assemble the isolated run directory (per variant -> distinct dir even with a shared STAMP).
    local RUN_DIR="$RUN_ROOT/occDrivaerRun${STAMP}${SUFFIX}-${MP_LABEL}"
    echo "================================================================"
    echo " [$MP_LABEL] building run dir: $RUN_DIR"
    build_run_dir "$RUN_DIR"

    # 1b. Pin the production window on THIS run dir's controlDict directly. The case-level override
    #     (lines ~94-97) is reverted by the harness between variants (the STEPS=30 pin / per-run dict
    #     restore), so build_run_dir would otherwise copy a 30-step controlDict for every variant after
    #     the first -- which is exactly why an earlier mp run left `float` at endTime=30 while `double`
    #     got 1000. Applying it per run dir makes each variant march the full 0 -> PROD_END_TIME.
    foamDictionary -entry stopAt        -set endTime                -disableFunctionEntries "$RUN_DIR/system/controlDict" >/dev/null
    foamDictionary -entry endTime       -set "$PROD_END_TIME"       -disableFunctionEntries "$RUN_DIR/system/controlDict" >/dev/null
    foamDictionary -entry writeControl  -set "$PROD_WRITE_CONTROL"  -disableFunctionEntries "$RUN_DIR/system/controlDict" >/dev/null
    foamDictionary -entry writeInterval -set "$PROD_WRITE_INTERVAL" -disableFunctionEntries "$RUN_DIR/system/controlDict" >/dev/null

    # 2. Install the chosen p-solver fvSolution (configFile swap + cache keys), leaving the template's
    #    l1ScaledResidual/tolerance/relTol intact. sed on the configFile line -- NOT foamDictionary.
    if [ "$BEST_CACHE" = "true" ]; then
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_CFG;\n        cacheSolver      true;\n        preconditionerRebuildInterval $BEST_REBUILD;#" \
            "$MG_TEMPLATE" > "$RUN_DIR/system/fvSolution"
    else
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$BEST_CFG;#" "$MG_TEMPLATE" > "$RUN_DIR/system/fvSolution"
    fi

    # 3. Forces/monitoring into the run dir's controlDict (already carries the production window).
    add_probes "$RUN_DIR/system/controlDict"

    # 4. Run in the run dir (subshell keeps the main shell in the case dir for the restore trap).
    #    reset_to_t0 runs here on the REAL processor*/ dirs, clearing old time dirs so each variant
    #    marches a FRESH 0 -> endTime; the subshell writes fresh ones through the processor symlinks.
    local log="$RUN_DIR/${name}-${STAMP}.log"
    : > "$log"
    ln -sf "$log" "$RESULTS/${name}-${STAMP}.log"
    RUN_LOG["$name"]="$log"
    echo "================================================================"
    echo " $name : PRODUCTION p = Ginkgo LOCALIZED MG ($BEST_CFG, ${MP_LABEL} precond), cache=${BEST_CACHE}, rebuildInterval=${BEST_REBUILD}, endTime=${PROD_END_TIME}${KEYWORD:+, keyword=${KEYWORD}}"
    echo "   run dir -> $RUN_DIR"
    echo "   log     -> $log"
    echo "================================================================"
    reset_to_t0
    local t0=$SECONDS
    ( cd "$RUN_DIR" && kokkos_launch "$BIN" ) > "$log" 2>&1 &
    local pid=$!
    tail_window "$pid" "$log"
    wait "$pid"; local rc=$?
    collect_kokkos_output "$log" "$RUN_DIR"
    echo "   exit=$rc  wall=$((SECONDS - t0))s  steps=$(grep -c '^Time = ' "$log")"
    RUNS+=("$name")
}

# Run each precision variant back to back: bake the CG-10 coarse solve into its config, then march it.
for variant in "${MP_VARIANTS[@]}"; do
    MP_LABEL="${variant%%:*}"
    BEST_CFG="${variant#*:}"
    if [ ! -f "system/gko/$BEST_CFG" ]; then
        echo "!! [$MP_LABEL] config 'system/gko/$BEST_CFG' not found -- skipping variant"; continue
    fi
    if [ "$BEST_COARSE_SOLVER" != "off" ]; then
        BEST_CFG="$(bake_coarse_solver "$BEST_CFG")"
        echo "   [$MP_LABEL] MG coarsest_solver = ${BEST_COARSE_SOLVER}${BEST_COARSE_ITERS}  ->  $BEST_CFG"
    fi
    run_production
done

print_summary
report_cache_reuse
