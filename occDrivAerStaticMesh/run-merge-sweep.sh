#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
# 1) wait for the running localized-nosc fill to finish (don't contend for GPUs, don't rebuild under it)
while ! grep -q "localized nosc fill done" "$OUT/locnoscfill.output" 2>/dev/null; do sleep 30; done
echo "=== fill done; rebuilding for neon::pgmMerge1 registration (additive header change) ==="
# 2) rebuild NeoN (ginkgo.hpp header change adds pgmMerge1 to the registry; additive, existing configs
#    produce identical results). This is the ONE allowed rebuild -- it happens BEFORE the merge sweep,
#    never during it.
cd /storage/home/greole/code/NeoFOAM/build/profilingnvidia_h200
module load gcc/13.3.0 cuda/13.0.2 cmake 2>/dev/null
ninja neoSimpleFoam 2>&1 | grep -E 'error:|FAILED|Linking CXX executable' | tail -4
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
# 3) sanity: does pgmMerge1 actually load? one throwaway 2-step run at global no-sc L4/merge1.
echo "=== pgmMerge1 smoke test (2 steps) ==="
MERGE=1 SC=false LEVELS="4" TOLS="0.1" STEPS=2 STUDY_SUFFIX=-mergesmoke ./phase3h-paper-study-mgsc-coarse-reltol.sh 2>&1 | tail -3
# 4) the sweep: 4 configs x max_levels{2,4,6,8} x merge{1,2,3}, tol fixed 0.1. 48 cells.
for M in 1 2 3; do
  echo "=== MERGE=$M ==="
  MERGE=$M SC=false LEVELS="2 4 6 8" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3h-paper-study-mgsc-coarse-reltol.sh  # global no-sc
  MERGE=$M SC=post  LEVELS="2 4 6 8" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3h-paper-study-mgsc-coarse-reltol.sh  # global sc-post
  MERGE=$M SC=none  LEVELS="2 4 6 8" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3i-paper-study-localized-reltol.sh    # localized no-sc
  MERGE=$M SC=post  LEVELS="2 4 6 8" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3i-paper-study-localized-reltol.sh    # localized sc-post
done
echo "merge sweep done"
