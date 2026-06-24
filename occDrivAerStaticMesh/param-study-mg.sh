#!/bin/bash
#
# Multigrid-pressure parameter study for occDrivAerStaticMesh.
#
# Sweeps the Ginkgo multigrid p-solver over system/gko/p-multigrid.L*.sc*.json
# (8 variants: max_levels {2,4,10,15} x scale_correction {0,2}), regenerated from
# system/gko/p-multigrid.json by gen-mg-variants.py at startup. Each run reuses
# fvSolution.case3 as a template and only swaps the p-solver configFile path;
# U/k/omega stay on smoothSolver/GaussSeidel. Runs/summary are named
# pMG-L<levels>-sc<scale>-ukoSmooth.
#
# A turbulence-free baseline (pMGbase-laminar: same multigrid p-solver, simulationType
# laminar) is appended to the full sweep and can be requested on its own.
#
# Usage:   ./param-study-mg.sh                 # full sweep (all variants) + laminar
#          ./param-study-mg.sh L4.sc2          # a single variant
#          ./param-study-mg.sh L4.sc2 L10.sc0  # selected variants (mg- prefix optional)
#          ./param-study-mg.sh laminar         # only the laminar baseline
#
# Shared env / run-window pinning / run_one / summary: param-study-common.sh.

source "$(dirname "$0")/param-study-common.sh"

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

run_mg_variant() {
    # $1 = variant json basename, e.g. p-multigrid.L4.sc2.json
    local json="$1"
    if [ ! -f "system/gko/$json" ]; then
        echo "!! no MG variant 'system/gko/$json' — skipping"; return
    fi
    local tag="${json#p-multigrid.}"; tag="${tag%.json}"   # L4.sc2
    local lev="${tag#L}"; lev="${lev%%.*}"                 # 4
    local sc="${tag##*sc}"                                 # 2
    # Reuse case3's U/k/omega block; only swap the p-solver configFile path.
    # Anchor to the indented entry line so the "configFile" mentions in the
    # template's comment header are left untouched.
    sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$json;#" "$MG_TEMPLATE" > "$TMP_FVSOL"
    run_one "pMG-L${lev}-sc${sc}-ukoSmooth" "$TMP_FVSOL" \
        "p = Ginkgo MG (max_levels=$lev, scale_correction=$sc), U/k/omega=smoothSolver/GS"
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

ensure_mg_variants

VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
    # Full sweep over every generated variant, plus the laminar baseline.
    found=0
    for j in system/gko/p-multigrid.L*.sc*.json; do
        [ -f "$j" ] || continue
        run_mg_variant "$(basename "$j")"
        found=1
    done
    [ "$found" -eq 0 ] && echo "!! no MG variant files found under system/gko/"
    run_laminar
else
    # Selected runs; accept "L4.sc2", "mg-L4.sc2", or "laminar".
    for v in "${VARIANTS[@]}"; do
        if [ "$v" = "laminar" ]; then
            run_laminar
        else
            run_mg_variant "p-multigrid.${v#mg-}.json"
        fi
    done
fi

print_summary
