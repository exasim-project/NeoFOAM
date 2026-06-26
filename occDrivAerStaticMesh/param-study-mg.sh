#!/bin/bash
#
# Multigrid-pressure parameter study for occDrivAreStaticMesh.
#
# Sweeps the Ginkgo multigrid p-solver over system/gko/p-multigrid.L*.sc*.json
# (8 variants: max_levels {2,4,10,15} x scale_correction {0,2}), regenerated from
# system/gko/p-multigrid.json by gen-mg-variants.py at startup. Each run reuses
# fvSolution.case3 as a template and only swaps the p-solver configFile path;
# U/k/omega stay on smoothSolver/GaussSeidel. Runs/summary are named
# pMG-L<levels>-sc<scale>-ukoSmooth.
#
# The full sweep also appends three extra runs (also requestable on their own):
#   fcg       pFCG-MGprec-ukoSmooth : FCG outer solver + Multigrid preconditioner
#             (system/gko/p-fcg-multigrid.json)
#   mgsolver  pMGsolver-ukoSmooth         : Multigrid as the global solver, no outer Krylov
#             (system/gko/p-multigrid-solver.json)
#   directcoarse pMG-directcoarse-ukoSmooth: Cg+MG with a direct (LU) coarsest solver
#             (system/gko/p-multigrid-directcoarse.json)
#   smooth2   pMG-smooth2-ukoSmooth       : Cg+MG with a 2-iteration Jacobi V-cycle smoother
#             (system/gko/p-multigrid-smooth2.json)
#   cgcoarse  pMG-cgcoarse-ukoSmooth      : Cg+MG with a CG+Jacobi coarsest solver
#             (system/gko/p-multigrid-cgcoarse.json)
#   native    native-simpleFoam           : native OpenFOAM simpleFoam, GAMG p-solver
#             (system/paramStudy/fvSolution.native) — the non-NeoFOAM reference run
#   native-pcg native-pcg-simpleFoam       : native OpenFOAM simpleFoam, PCG/diagonal p-solver
#             (system/paramStudy/fvSolution.native-pcg)
#   laminar   pMGbase-laminar             : base multigrid p-solver, simulationType laminar
# The fcg/mgsolver/directcoarse/smooth2/cgcoarse configs are derived from p-multigrid.json by gen-mg-variants.py.
# (The SELL-P matrix-format study lives in param-study.sh: Sellp only applies to the plain
#  Ginkgo CG solver, not the multigrid sweep here.)
#
# Usage:   ./param-study-mg.sh                 # full sweep + fcg + mgsolver + directcoarse + smooth2 + cgcoarse + native + laminar
#          ./param-study-mg.sh L4.sc2          # a single variant
#          ./param-study-mg.sh L4.sc2 L10.sc0  # selected variants (mg- prefix optional)
#          ./param-study-mg.sh fcg mgsolver    # the FCG and standalone-Multigrid runs
#          ./param-study-mg.sh directcoarse    # Cg+MG with direct coarsest solver
#          ./param-study-mg.sh smooth2         # Cg+MG with a 2-iteration Jacobi smoother
#          ./param-study-mg.sh cgcoarse        # Cg+MG with a CG+Jacobi coarsest solver
#          ./param-study-mg.sh native          # native OpenFOAM simpleFoam (GAMG) baseline
#          ./param-study-mg.sh native-pcg      # native OpenFOAM simpleFoam (PCG/diagonal) baseline
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
    # $1 = variant json basename, e.g. p-multigrid.L4.sc2.json
    local json="$1"
    local tag="${json#p-multigrid.}"; tag="${tag%.json}"   # L4.sc2
    local lev="${tag#L}"; lev="${lev%%.*}"                 # 4
    local sc="${tag##*sc}"                                 # 2
    run_pconfig "pMG-L${lev}-sc${sc}-ukoSmooth" "$json" \
        "p = Ginkgo MG (max_levels=$lev, scale_correction=$sc), U/k/omega=smoothSolver/GS"
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
    run_fcg          # FCG + Multigrid preconditioner
    run_mgsolver     # Multigrid as the global solver
    run_directcoarse # Cg+MG with direct (LU) coarsest solver
    run_smooth2      # Cg+MG with a 2-iteration Jacobi smoother
    run_cgcoarse     # Cg+MG with a CG+Jacobi coarsest solver
    run_native       # native OpenFOAM simpleFoam (GAMG) baseline
    run_native_pcg   # native OpenFOAM simpleFoam (PCG/diagonal) baseline
    run_laminar
else
    # Selected runs; accept "L4.sc2", "mg-L4.sc2", "fcg", "mgsolver",
    # "directcoarse", "smooth2", "cgcoarse", "native", "native-pcg", or "laminar".
    for v in "${VARIANTS[@]}"; do
        case "$v" in
            laminar)      run_laminar ;;
            fcg)          run_fcg ;;
            mgsolver)     run_mgsolver ;;
            directcoarse) run_directcoarse ;;
            smooth2)      run_smooth2 ;;
            cgcoarse)     run_cgcoarse ;;
            native)       run_native ;;
            native-pcg)   run_native_pcg ;;
            *)            run_mg_variant "p-multigrid.${v#mg-}.json" ;;
        esac
    done
fi

print_summary
