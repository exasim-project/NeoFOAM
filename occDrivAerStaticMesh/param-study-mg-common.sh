#!/bin/bash
#
# Shared multigrid-pressure run helpers for the occDrivAre parameter studies. SOURCED by
# param-study-mg.sh (the headline MG-usage comparison) and param-study-mg-tuning.sh (the
# max_levels sweep + smoother/coarse-solver/outer-Krylov tuning variants) -- not run directly.
#
# Provides: MG variant regeneration (ensure_mg_variants), the configFile-swap run helper
# (run_pconfig), the level-sweep run (run_mg_variant), one run_* per named variant, and a
# run_named dispatcher that maps a keyword to its run_* function. Every variant config under
# system/gko/ is derived from system/gko/p-multigrid.json by gen-mg-variants.py.
#
# Shared env / run-window pinning / run_one / summary come from param-study-common.sh, which
# this file sources (and which cd's into the case directory, so the relative paths below work).

source "$(dirname "${BASH_SOURCE[0]}")/param-study-common.sh"

MGGEN="system/gko/gen-mg-variants.py"
MG_TEMPLATE="$CFGDIR/fvSolution.case3"

_mg_generated=0
ensure_mg_variants() {
    [ "$_mg_generated" -eq 1 ] && return 0
    if python3 "$MGGEN" >/dev/null 2>&1; then
        echo "   (regenerated MG variants from system/gko/p-multigrid.json)"
    else
        echo "!! $MGGEN failed — using existing variant files"
    fi
    _mg_generated=1
}

run_pconfig() {
    # $1 = run name   $2 = p-solver json basename (in system/gko/)   $3 = description
    # Reuse case3's U/k/omega block; only swap the p-solver configFile path. Anchor to
    # the indented entry line so the "configFile" mentions in the template's comment
    # header are left untouched.
    local name="$1" json="$2" desc="$3"
    if [ ! -f "system/gko/$json" ]; then
        echo "!! no p-config 'system/gko/$json' — skipping"; return
    fi
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$json;#" "$MG_TEMPLATE" > "$TMP_FVSOL"
    run_one "$name" "$TMP_FVSOL" "$desc"
}

run_mg_variant() {
    # $1 = variant json basename, e.g. p-multigrid.L4.sc0.json
    local json="$1"
    local tag="${json#p-multigrid.}"; tag="${tag%.json}"   # L4.sc0
    local lev="${tag#L}"; lev="${lev%%.*}"                 # 4
    local sc="${tag##*sc}"                                 # 0
    run_pconfig "pMG-L${lev}-sc${sc}-ukoSmooth" "$json" \
        "p = Ginkgo MG (max_levels=$lev, scale_correction=$sc), U/k/omega=smoothSolver/GS"
}

run_base() {
    # PCG (Cg) outer solver with the Multigrid block as preconditioner -- the canonical base
    # config (system/gko/p-multigrid.json, max_levels=10). The headline "PCG + Multigrid as a
    # preconditioner" run.
    run_pconfig "pMG-ukoSmooth" "p-multigrid.json" \
        "p = Ginkgo Cg + Multigrid preconditioner (base), U/k/omega=smoothSolver/GS"
}

run_fcg() {
    # FCG (flexible CG) outer solver with the Multigrid block as preconditioner.
    run_pconfig "pFCG-MGprec-ukoSmooth" "p-fcg-multigrid.json" \
        "p = Ginkgo FCG + Multigrid preconditioner, U/k/omega=smoothSolver/GS"
}

run_mgsolver() {
    # Multigrid as the global solver (no outer Krylov): V-cycles to convergence.
    run_pconfig "pMGsolver-ukoSmooth" "p-multigrid-solver.json" \
        "p = Ginkgo Multigrid as global solver, U/k/omega=smoothSolver/GS"
}

run_directcoarse() {
    # Cg + Multigrid, but the coarsest level solved exactly by a direct LU solver.
    run_pconfig "pMG-directcoarse-ukoSmooth" "p-multigrid-directcoarse.json" \
        "p = Ginkgo Cg+MG, direct (LU) coarsest solver, U/k/omega=smoothSolver/GS"
}

run_smooth2() {
    # Cg + Multigrid with a stronger V-cycle smoother: 2 Ir/Jacobi iterations per
    # pre/post sweep instead of the base's weak single iteration.
    run_pconfig "pMG-smooth2-ukoSmooth" "p-multigrid-smooth2.json" \
        "p = Ginkgo Cg+MG, 2-iteration Jacobi smoother, U/k/omega=smoothSolver/GS"
}

run_cgcoarse() {
    # Cg + Multigrid, but the coarsest level solved by an inner Jacobi-preconditioned CG.
    run_pconfig "pMG-cgcoarse-ukoSmooth" "p-multigrid-cgcoarse.json" \
        "p = Ginkgo Cg+MG, CG+Jacobi coarsest solver, U/k/omega=smoothSolver/GS"
}

run_localized() {
    # LOCALIZED multigrid preconditioner: the whole Multigrid runs on each rank's local
    # block, coupled by one outer restricted-additive Schwarz, with a plain local-Jacobi
    # (localized) smoother. Contrast with the distributed base (MG on the distributed matrix,
    # distributed Schwarz{Jacobi} smoothers). Config-file analog of OGL type_=="Schwarz".
    run_pconfig "pMG-localized-ukoSmooth" "p-multigrid-localized.json" \
        "p = Ginkgo Cg + LOCALIZED MG (Schwarz{MG(local)}, local-Jacobi smoother), U/k/omega=smoothSolver/GS"
}

run_localized_level() {
    # $1 = localized level json basename, e.g. p-multigrid-localized.L8.json. The LOCALIZED
    # multigrid preconditioner (run_localized's Schwarz{MG(local)} + local-Jacobi smoother) swept
    # over the inner Multigrid's max_levels. Drives the localized max_levels study.
    local json="$1"
    local lev="${json#p-multigrid-localized.L}"; lev="${lev%.json}"   # 8
    run_pconfig "pMG-localized-L${lev}-ukoSmooth" "$json" \
        "p = Ginkgo Cg + LOCALIZED MG (max_levels=$lev, Schwarz{MG(local)}, local-Jacobi smoother), U/k/omega=smoothSolver/GS"
}

run_localized_level_sweep() {
    # Run the whole LOCALIZED max_levels sweep: every generated p-multigrid-localized.L*.json
    # (levels 2,4,6,8,10,15,20 from gen-mg-variants.py), in ascending level order.
    local found=0 j
    for j in $(ls -v system/gko/p-multigrid-localized.L*.json 2>/dev/null); do
        [ -f "$j" ] || continue
        run_localized_level "$(basename "$j")"
        found=1
    done
    [ "$found" -eq 0 ] && echo "!! no localized level-variant files found under system/gko/"
}

run_localized_solver() {
    # The localized multigrid promoted to the GLOBAL solver (no outer Krylov): the outer
    # Schwarz{Multigrid(local)} block V-cycles to convergence on the base's outer criteria.
    run_pconfig "pMG-localized-solver-ukoSmooth" "p-multigrid-localized-solver.json" \
        "p = Ginkgo LOCALIZED MG as global solver (Schwarz{MG(local)}), U/k/omega=smoothSolver/GS"
}

run_scalecorr() {
    # SCALE-CORRECTED multigrid: outer solver::Ir (scale_correction="backward") wrapping a
    # solver::Multigrid with per-level scale_correction=true, one V-cycle per outer iteration.
    # NOTE: the scale_correction config keys ship in NeoN_GINKGO_TAG (241deca), which is the
    # PRODUCTION build -- the param-study default (NEON_BUILD=production in param-study-common.sh),
    # so this variant runs as-is. The profiling build is on an older ginkgo and ABORTS on the
    # MG-level "scale_correction" key, so do NOT run this with NEON_BUILD=profiling.
    run_pconfig "pMG-scalecorr-ukoSmooth" "p-multigrid-scalecorr.json" \
        "p = Ginkgo scale-corrected MG (Ir[backward] + MG[scale_correction]), U/k/omega=smoothSolver/GS"
}

run_scalecorr_localized() {
    # SCALE-CORRECTED + LOCALIZED multigrid: the scalecorr construction (outer solver::Ir
    # scale_correction="backward" wrapping a solver::Multigrid scale_correction=true) but with
    # the inner Multigrid run on each rank's LOCAL block, coupled by one outer restricted-additive
    # Schwarz, with a plain local-Jacobi smoother -- the localized smoother. Combines the
    # scalecorr Rayleigh correction with the localized per-rank V-cycle.
    # NOTE: needs the scale_correction config keys from NeoN_GINKGO_TAG (241deca) -- the
    # PRODUCTION build (study default), like run_scalecorr; do NOT run with NEON_BUILD=profiling.
    run_pconfig "pMG-scale-correction-localized-ukoSmooth" "p-multigrid-scalecorr-localized.json" \
        "p = Ginkgo scale-corrected LOCALIZED MG (Ir[backward] + Schwarz{MG[scale_correction](local)}), U/k/omega=smoothSolver/GS"
}

# ------------------------------------------------- preconditioner-cache reuse sweep
# preconditionerRebuildInterval values swept by run_cache_sweep / the "cache-sweep" keyword.
CACHE_INTERVALS=(2 5 10 20 50 100)

# Preconditioner configs swept by the cache study, as "label:configFile.json". Each is run with
# cacheSolver=true at every CACHE_INTERVALS value:
#   base       Cg + distributed Multigrid preconditioner          -> config(Cg+Multigrid)
#   localized  Cg + Schwarz{ Multigrid(local) } preconditioner     -> config(Cg+Schwarz(Multigrid))
#   scalecorr  Ir(scale_correction) { solver: Multigrid }          -> config(Ir+Multigrid)
# update_matrix_value reuse targets, per config: the bound Multigrid preconditioner (base); the
# per-rank Multigrid inside Schwarz (localized, via the Schwarz UpdateMatrixValue patch); the inner
# Multigrid of the outer Ir (scalecorr, via the Ir-inner-solver branch in cacheOrUpdateSolver).
# Override the set by exporting CACHE_VARIANTS="label:file.json ..." before the run.
CACHE_VARIANTS=(${CACHE_VARIANTS:-
    "base:p-multigrid.json"
    "localized:p-multigrid-localized.json"
    "scalecorr:p-multigrid-scalecorr.json"})

run_cache_interval() {
    # $1 = variant label   $2 = configFile basename   $3 = preconditionerRebuildInterval.
    # Run the named preconditioner config with solver caching ON (cacheSolver true): the generated
    # solver + its multigrid hierarchy are cached and, on later pressure solves with unchanged matrix
    # structure, refreshed in place via gko::UpdateMatrixValue::update_matrix_value (reusing the
    # expensive Pgm aggregation) instead of being rebuilt from scratch. preconditionerRebuildInterval
    # forces a full rebuild every Nth solve so aggregation drift across the steady iteration stays
    # bounded. The two keys are injected into the p{} solver block of the base case3 template by sed
    # (NOT foamDictionary, which re-tokenises the "system/gko/..." path into "system / gko / ...").
    #
    # CAVEAT: the localized (and to a lesser degree scalecorr) preconditioner is more sensitive to
    # aggregation staleness than the distributed base -- large rebuild intervals can degrade
    # convergence (and on a stiff case diverge), so the small end of the sweep is the useful range
    # there. report_cache_reuse tallies the per-solve "[GinkgoSolver] p-cache:" diagnostics:
    # "rebuild(generate)" on solve 1 and every Nth solve, "reuse(update_matrix_value)" otherwise;
    # an all-"rebuild" row => that config's reuse path did not engage in the running binary.
    local label="$1" cfg="$2" n="$3"
    local name="pMG-${label}-cache-rebuild${n}"
    if [ ! -f "system/gko/$cfg" ]; then
        echo "!! no cache config 'system/gko/$cfg' -- skipping $name"; return
    fi
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$cfg;\n        cacheSolver      true;\n        preconditionerRebuildInterval $n;#" \
        "$MG_TEMPLATE" > "$TMP_FVSOL"
    run_one "$name" "$TMP_FVSOL" \
        "p = Ginkgo $label MG ($cfg), cacheSolver=true, preconditionerRebuildInterval=$n"
}

run_cache_variant() {
    # $1 = variant label (base|localized|scalecorr). Sweep one config over all CACHE_INTERVALS.
    ensure_mg_variants
    local label="$1" v cfg="" n
    for v in "${CACHE_VARIANTS[@]}"; do
        [ "${v%%:*}" = "$label" ] && cfg="${v#*:}"
    done
    if [ -z "$cfg" ]; then echo "!! unknown cache variant '$label'"; return; fi
    for n in "${CACHE_INTERVALS[@]}"; do
        run_cache_interval "$label" "$cfg" "$n"
    done
}

run_cache_sweep() {
    # Full study: every CACHE_VARIANTS config x every CACHE_INTERVALS value, each a single
    # STEPS-iteration run with solver caching enabled.
    ensure_mg_variants
    local v label cfg n
    for v in "${CACHE_VARIANTS[@]}"; do
        label="${v%%:*}"; cfg="${v#*:}"
        for n in "${CACHE_INTERVALS[@]}"; do
            run_cache_interval "$label" "$cfg" "$n"
        done
    done
}

report_cache_reuse() {
    # Tally the per-solve "[GinkgoSolver] p-cache:" diagnostics for every cache run in RUNS, so the
    # study proves the preconditioner is actually cached and reused (reuses should dominate, with one
    # rebuild every preconditionerRebuildInterval solves). A row of 0/0 means the diagnostic never
    # fired -> caching did not engage for that config.
    local any=0 c
    for c in "${RUNS[@]}"; do
        case "$c" in pMG-*-cache-rebuild*) any=1; break ;; esac
    done
    [ "$any" -eq 0 ] && return 0
    echo
    echo "###################### PRECONDITIONER CACHE REUSE ######################"
    printf "%-32s %9s %9s %9s\n" run rebuilds reuses "reuse%"
    local log rb ru tot pct
    for c in "${RUNS[@]}"; do
        case "$c" in pMG-*-cache-rebuild*) ;; *) continue ;; esac
        log="${RUN_LOG[$c]}"
        [ -n "$log" ] && [ -f "$log" ] || continue
        rb=$(grep -c 'p-cache: rebuild' "$log" 2>/dev/null); rb=${rb:-0}
        ru=$(grep -c 'p-cache: reuse'   "$log" 2>/dev/null); ru=${ru:-0}
        tot=$((rb + ru))
        if [ "$tot" -gt 0 ]; then pct=$(awk "BEGIN{printf \"%.0f\", 100*$ru/$tot}"); else pct="-"; fi
        printf "%-32s %9s %9s %9s\n" "$c" "$rb" "$ru" "$pct"
    done
    echo "#######################################################################"
    echo "(0/0 row => cacheSolver never engaged; reuse% high => update_matrix_value path exercised)"
}

run_native() {
    # Native-OpenFOAM baseline: real simpleFoam (NOT neoSimpleFoam) with a GAMG
    # p-solver and smoothSolver U/k/omega, from system/paramStudy/fvSolution.native.
    # Reference point for the whole Ginkgo multigrid-pressure sweep.
    local nat="$CFGDIR/fvSolution.native"
    if [ ! -f "$nat" ]; then
        echo "!! no native fvSolution '$nat' — skipping"; return
    fi
    run_one "native-simpleFoam" "$nat" \
        "p = OpenFOAM GAMG/GS, U/k/omega = smoothSolver/GS (native simpleFoam)" \
        "simpleFoam"
}

run_native_pcg() {
    # Native-OpenFOAM baseline with a PCG/diagonal p-solver (the PCG counterpart of the
    # GAMG run_native), from system/paramStudy/fvSolution.native-pcg, via real simpleFoam.
    local nat="$CFGDIR/fvSolution.native-pcg"
    if [ ! -f "$nat" ]; then
        echo "!! no native fvSolution '$nat' — skipping"; return
    fi
    run_one "native-pcg-simpleFoam" "$nat" \
        "p = OpenFOAM PCG/diagonal, U/k/omega = smoothSolver/GS (native simpleFoam)" \
        "simpleFoam"
}

run_laminar() {
    # Same multigrid p-solver (base p-multigrid.json via the case3 template) but with
    # the turbulence model switched OFF (simulationType laminar -> NeoFOAM Laminar:
    # nut=0, nuEff=nu). Isolates the pressure/momentum coupling from kOmegaSST and
    # serves as a turbulence-free baseline for the multigrid-pressure comparison.
    # turbulenceProperties is toggled here and restored right after (the exit trap in
    # param-study-common.sh is the safety net if the run is interrupted).
    foamDictionary -entry simulationType -set laminar \
        -disableFunctionEntries constant/turbulenceProperties >/dev/null
    run_one "pMGbase-laminar" "$MG_TEMPLATE" "p = Ginkgo MG (base), turbulence = laminar"
    foamDictionary -entry simulationType -set RAS \
        -disableFunctionEntries constant/turbulenceProperties >/dev/null
}

run_named() {
    # Dispatch a single variant keyword to its run_* function. A bare "L<lev>.sc<sc>" (with an
    # optional "mg-" prefix) is treated as a level-sweep variant. Shared by both top scripts so
    # either can run any variant explicitly; they differ only in their no-argument defaults.
    case "$1" in
        base|pcg-mg)      run_base ;;
        localized)        run_localized ;;
        mgsolver)         run_mgsolver ;;
        scalecorr)        run_scalecorr ;;
        scalecorr-localized|scale-correction-localized) run_scalecorr_localized ;;
        fcg)              run_fcg ;;
        directcoarse)     run_directcoarse ;;
        smooth2)          run_smooth2 ;;
        cgcoarse)         run_cgcoarse ;;
        localized-solver) run_localized_solver ;;
        localized-levels) run_localized_level_sweep ;;
        cache-sweep)      run_cache_sweep ;;
        cache-base|cache-localized|cache-scalecorr) run_cache_variant "${1#cache-}" ;;
        localized-L*|locL*) run_localized_level "p-multigrid-localized.L${1##*L}.json" ;;
        native)           run_native ;;
        native-pcg)       run_native_pcg ;;
        laminar)          run_laminar ;;
        *)                run_mg_variant "p-multigrid.${1#mg-}.json" ;;
    esac
}
