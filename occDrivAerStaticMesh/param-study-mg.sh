#!/bin/bash
#
# Multigrid-pressure parameter study for occDrivAreStaticMesh -- HEADLINE comparison.
#
# Four multigrid USAGE modes, each a single STEPS-iteration run from time 0 (apples-to-apples):
#   base      pMG-ukoSmooth             : PCG (Cg) + Multigrid as a PRECONDITIONER -- the
#             canonical base config (system/gko/p-multigrid.json, max_levels=10).
#   localized pMG-localized-ukoSmooth   : PCG + LOCALIZED Multigrid preconditioner -- the whole
#             Multigrid runs on each rank's local block, coupled by one outer Schwarz, with a
#             local-Jacobi smoother (Schwarz{MG(local)}); the config-file analog of OGL
#             type_=="Schwarz" (system/gko/p-multigrid-localized.json).
#   mgsolver  pMGsolver-ukoSmooth       : Multigrid as the TOP-LEVEL solver, no outer Krylov --
#             V-cycles to convergence (system/gko/p-multigrid-solver.json).
#   scalecorr pMG-scalecorr-ukoSmooth   : Multigrid with SCALE CORRECTION -- outer solver::Ir
#             (scale_correction="backward") wrapping a solver::Multigrid with per-level
#             scale_correction=true, one V-cycle/iter (system/gko/p-multigrid-scalecorr.json).
#             Needs the scale_correction config keys from NeoN_GINKGO_TAG (241deca) -- the
#             PRODUCTION build, which is the study default (NEON_BUILD=production), so it runs
#             as-is. Do NOT run it with NEON_BUILD=profiling (older ginkgo aborts on the key).
#   sclocal   pMG-scale-correction-localized-ukoSmooth : the scalecorr construction made
#             LOCALIZED -- outer solver::Ir(scale_correction="backward") wrapping a
#             Schwarz{Multigrid(local, scale_correction=true)} with a local-Jacobi smoother
#             (system/gko/p-multigrid-scalecorr-localized.json). Combines scalecorr's Rayleigh
#             correction with localized's per-rank V-cycle. Same PRODUCTION-build requirement
#             as scalecorr -- do NOT run with NEON_BUILD=profiling.
#
# The max_levels sweep and the smoother/coarse-solver/outer-Krylov TUNING variants (fcg,
# directcoarse, smooth2, cgcoarse, localized-solver) plus the native/laminar baselines live in
# param-study-mg-tuning.sh. Run helpers are shared via param-study-mg-common.sh.
#
# SOLVER-CACHE sweep (cache-sweep):
#   Sweeps the Ginkgo solver's preconditionerRebuildInterval over 2,5,10,20,50,100 with
#   cacheSolver=true, across THREE preconditioner configs: base (Cg + distributed Multigrid),
#   localized (Cg + Schwarz{Multigrid(local)}), and scalecorr (Ir(scale_correction){Multigrid}).
#   The generated solver + MG hierarchy are cached and reused across pressure solves via
#   update_matrix_value, rebuilt from scratch only every Nth solve. After the runs a PRECONDITIONER
#   CACHE REUSE table tallies the per-solve "[GinkgoSolver] p-cache: rebuild|reuse" log diagnostics
#   so you can confirm the preconditioner is actually cached and reused per config (reuse% high; a
#   0/0 row means that config's reuse path did not engage). The localized/scalecorr preconditioners
#   are more staleness-sensitive, so their useful range is the small-interval end.
#
# Usage:   ./param-study-mg.sh                 # all four headline runs
#          ./param-study-mg.sh localized       # a single headline run
#          ./param-study-mg.sh base scalecorr  # selected headline runs
#          ./param-study-mg.sh mgsolver        # Multigrid as the top-level solver
#          ./param-study-mg.sh cache-sweep     # {base,localized,scalecorr} x interval{2,5,10,20,50,100}
#          ./param-study-mg.sh cache-localized # one cache config across all intervals
#          (any tuning keyword also works here, e.g. ./param-study-mg.sh L4.sc0 -- see
#           param-study-mg-tuning.sh)

STUDY_TYPE="${STUDY_TYPE:-mg-headline}"
source "$(dirname "$0")/param-study-mg-common.sh"

ensure_mg_variants

VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(base localized mgsolver scalecorr scalecorr-localized)
for v in "${VARIANTS[@]}"; do
    run_named "$v"
done

print_summary
report_cache_reuse
