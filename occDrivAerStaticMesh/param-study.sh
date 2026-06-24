#!/bin/bash
#
# Parameter study for occDrivAerStaticMesh — linear-solver configurations.
# Runs STEPS SIMPLE iterations for each case, every case starting from time 0 so
# the comparison is apples-to-apples.
#
#   case1   p: PCG / diagonal           | U,k,omega: smoothSolver / GaussSeidel
#   case2   p: PCG / diagonal           | U,k,omega: PBiCGStab    / diagonal
#   case3   p: configFile (multigrid)   | U,k,omega: smoothSolver / GaussSeidel
#
# The Ginkgo multigrid-pressure SWEEP lives in param-study-mg.sh (case3 is the
# single multigrid baseline kept here for reference against the PCG variants).
#
# Each caseN solver block lives in system/paramStudy/fvSolution.caseN; this
# script swaps it into system/fvSolution, runs, and restores the original.
# Logs/summary use self-documenting names (p-solver + U/k/omega solver):
#   case1 -> pPCG-ukoSmooth   case2 -> pPCG-ukoPBiCGStab   case3 -> pMGbase-ukoSmooth
#
# Usage:   ./param-study.sh            # all three cases
#          ./param-study.sh case2      # a single case
#
# Shared env / run-window pinning / run_one / summary: param-study-common.sh.

source "$(dirname "$0")/param-study-common.sh"

# Self-documenting log/summary name for each caseN (p-solver + U/k/omega solver).
case_label() {
    case "$1" in
        case1) echo "pPCG-ukoSmooth"    ;;  # p=PCG/diag, U/k/omega=smoothSolver/GS
        case2) echo "pPCG-ukoPBiCGStab" ;;  # p=PCG/diag, U/k/omega=PBiCGStab/diag
        case3) echo "pMGbase-ukoSmooth" ;;  # p=Ginkgo MG base, U/k/omega=smoothSolver/GS
        *)     echo "$1"                ;;
    esac
}

CASES=("$@"); [ ${#CASES[@]} -eq 0 ] && CASES=(case1 case2 case3)
for c in "${CASES[@]}"; do
    cfg="$CFGDIR/fvSolution.$c"
    if [ ! -f "$cfg" ]; then
        echo "!! no config for '$c' ($cfg) — skipping"; continue
    fi
    desc=$(sed -n 's#^// PARAM STUDY [^:]*: *##p' "$cfg" | head -1)
    run_one "$(case_label "$c")" "$cfg" "$desc"
done

print_summary
