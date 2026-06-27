#!/bin/bash
#
# Multigrid-pressure parameter study for occDrivAreStaticMesh -- TUNING sweep.
#
# The max_levels level study plus the smoother / coarse-solver / outer-Krylov tuning variants.
# The four HEADLINE usage modes (base PCG+MG, localized, mgsolver, scalecorr) live in
# param-study-mg.sh. Run helpers are shared via param-study-mg-common.sh; every variant config
# under system/gko/ is derived from system/gko/p-multigrid.json by gen-mg-variants.py.
#
# Level study (the default sweep), named pMG-L<levels>-sc<scale>-ukoSmooth:
#   system/gko/p-multigrid.L*.sc*.json   max_levels in {2,4,10,15}, scale_correction in {0}
#
# LOCALIZED level study (the default sweep), named pMG-localized-L<levels>-ukoSmooth:
#   system/gko/p-multigrid-localized.L*.json   max_levels in {2,4,6,8,10,15,20}
#   The localized MG (pMG-localized-ukoSmooth, Schwarz{MG(local)} + local-Jacobi smoother)
#   swept over the inner Multigrid's max_levels.
#
# Tuning variants (also in the default sweep; each requestable on its own):
#   fcg          pFCG-MGprec-ukoSmooth      : FCG outer solver + Multigrid preconditioner
#                (system/gko/p-fcg-multigrid.json)
#   directcoarse pMG-directcoarse-ukoSmooth : Cg+MG with a direct (LU) coarsest solver
#                (system/gko/p-multigrid-directcoarse.json)
#   smooth2      pMG-smooth2-ukoSmooth      : Cg+MG with a 2-iteration Jacobi V-cycle smoother
#                (system/gko/p-multigrid-smooth2.json)
#   cgcoarse     pMG-cgcoarse-ukoSmooth     : Cg+MG with a CG+Jacobi coarsest solver
#                (system/gko/p-multigrid-cgcoarse.json)
#   localized-solver pMG-localized-solver-ukoSmooth : the localized MG promoted to the GLOBAL
#                solver, no outer Krylov (system/gko/p-multigrid-localized-solver.json)
#   localized-levels pMG-localized-L<lev>-ukoSmooth : the LOCALIZED-MG max_levels sweep over
#                {2,4,6,8,10,15,20} (system/gko/p-multigrid-localized.L*.json). A single level is
#                requestable as localized-L<lev> (e.g. localized-L8).
#   laminar      pMGbase-laminar            : base multigrid p-solver, simulationType laminar
#
# Selectable-only (NOT in the default sweep; request explicitly):
#   native     native-simpleFoam     : native OpenFOAM simpleFoam, GAMG p-solver
#   native-pcg native-pcg-simpleFoam : native OpenFOAM simpleFoam, PCG/diagonal p-solver
#
# (The SELL-P matrix-format study lives in param-study.sh: Sellp only applies to the plain
#  Ginkgo CG solver, not the multigrid sweep here.)
#
# Usage:   ./param-study-mg-tuning.sh                 # level sweep + fcg + directcoarse + smooth2 + cgcoarse + localized-solver + laminar
#          ./param-study-mg-tuning.sh L4.sc0          # a single level variant
#          ./param-study-mg-tuning.sh L4.sc0 L10.sc0  # selected level variants (mg- prefix optional)
#          ./param-study-mg-tuning.sh fcg             # FCG + Multigrid preconditioner
#          ./param-study-mg-tuning.sh directcoarse    # Cg+MG with direct coarsest solver
#          ./param-study-mg-tuning.sh smooth2         # Cg+MG with a 2-iteration Jacobi smoother
#          ./param-study-mg-tuning.sh cgcoarse        # Cg+MG with a CG+Jacobi coarsest solver
#          ./param-study-mg-tuning.sh localized-solver # LOCALIZED MG as the global solver
#          ./param-study-mg-tuning.sh localized-levels # LOCALIZED MG max_levels sweep {2..20}
#          ./param-study-mg-tuning.sh localized-L8     # a single localized level variant
#          ./param-study-mg-tuning.sh native          # native OpenFOAM simpleFoam (GAMG) baseline
#          ./param-study-mg-tuning.sh native-pcg      # native OpenFOAM simpleFoam (PCG/diagonal) baseline
#          ./param-study-mg-tuning.sh laminar         # only the laminar baseline
#          (any headline keyword also works here, e.g. ./param-study-mg-tuning.sh base)

STUDY_TYPE="${STUDY_TYPE:-mg-tuning}"
source "$(dirname "$0")/param-study-mg-common.sh"

ensure_mg_variants

VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
    # Default tuning sweep: every generated level variant, then the parameter-tuning variants,
    # then the laminar baseline. native / native-pcg are excluded -- request them explicitly.
    found=0
    for j in system/gko/p-multigrid.L*.sc*.json; do
        [ -f "$j" ] || continue
        run_mg_variant "$(basename "$j")"
        found=1
    done
    [ "$found" -eq 0 ] && echo "!! no MG level-variant files found under system/gko/"
    run_localized_level_sweep  # LOCALIZED MG max_levels sweep {2,4,6,8,10,15,20}
    run_fcg              # FCG + Multigrid preconditioner
    run_directcoarse     # Cg+MG with direct (LU) coarsest solver
    run_smooth2          # Cg+MG with a 2-iteration Jacobi smoother
    run_cgcoarse         # Cg+MG with a CG+Jacobi coarsest solver
    run_localized_solver # LOCALIZED MG as the global solver
    run_laminar
else
    for v in "${VARIANTS[@]}"; do
        run_named "$v"
    done
fi

print_summary
