#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
# Re-run BOTH sc grids on the HOISTED binary. The previous grids were measured with ~37 synchronizing
# raw cudaMalloc/cudaFree pairs per solve (dzc's eps + the relaxation scalar), worth ~3.4% of p_ms --
# above the ~1% noise floor, so they are not comparable to the no-sc grids. Only sc cells are affected
# (sc=off never enters the scale blocks), so the no-sc grids stand and are NOT re-run.
# FORCE=1: the archived logs are gone, but FORCE makes the intent explicit and is harmless.
# DO NOT rebuild the binary while this runs.
echo "=== GLOBAL MG + sc(post), hoisted binary ==="
FORCE=1 SC=post LEVELS="2 3 4 5 6 8" TOLS="0.25 0.2 0.15 0.1 0.01 0.001" STEPS=50 ./phase3h-paper-study-mgsc-coarse-reltol.sh
echo "=== LOCALIZED Schwarz{MG} + sc(post), hoisted binary ==="
FORCE=1 SC=post LEVELS="2 3 4 5 6 8" TOLS="0.25 0.2 0.15 0.1 0.01 0.001" STEPS=50 ./phase3i-paper-study-localized-reltol.sh
echo "sc grids hoisted done"
