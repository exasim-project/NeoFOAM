#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
# Both grids on the FAITHFUL+FUSED port, sc = post-only (OpenFOAM's default; beats 'both' at 2.401 vs 2.465).
# max_levels 2..8 for BOTH branches -- the earlier localized grid used {4,6,8,10}, which was not
# comparable with global's {2..8}. 6 levels x 6 tols = 36 cells per branch.
echo "=== GLOBAL MG + sc(post), faithful port ==="
SC=post LEVELS="2 3 4 5 6 8" TOLS="0.25 0.2 0.15 0.1 0.01 0.001" STEPS=50 ./phase3h-paper-study-mgsc-coarse-reltol.sh
echo "=== LOCALIZED Schwarz{MG} + sc(post), faithful port ==="
SC=post LEVELS="2 3 4 5 6 8" TOLS="0.25 0.2 0.15 0.1 0.01 0.001" STEPS=50 ./phase3i-paper-study-localized-reltol.sh
echo "sc grids done"
