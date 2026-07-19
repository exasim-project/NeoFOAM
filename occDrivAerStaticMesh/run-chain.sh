#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks

# --- 1) wait for the running localized NO-SC fill ---
while ! grep -q "localized nosc fill done" "$OUT/locnoscfill.output" 2>/dev/null; do sleep 30; done

# --- 2) fill the 6 missing localizedsc (sc-post) cells on the CURRENT (pre-rebuild) binary, so they
#         are binary-consistent with the other 30 sc-post cells. run_one skips the 30 that exist. ---
echo "=== filling missing localizedsc sc-post cells (current binary) ==="
SC=post LEVELS="2 3 4 5 6 8" TOLS="0.25 0.2 0.15 0.1 0.01 0.001" STEPS=50 \
  ./phase3i-paper-study-localized-reltol.sh
echo "localizedsc fill done"

# --- 3) rebuild for neon::pgmMerge1 (additive; does not affect any sc-post cell above) ---
echo "=== rebuilding for pgmMerge1 ==="
cd /storage/home/greole/code/NeoFOAM/build/profilingnvidia_h200
module load gcc/13.3.0 cuda/13.0.2 cmake 2>/dev/null
ninja neoSimpleFoam 2>&1 | grep -E 'error:|FAILED|Linking CXX executable' | tail -4
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
echo "=== pgmMerge1 smoke test (2 steps) ==="
MERGE=1 SC=false LEVELS="4" TOLS="0.1" STEPS=2 STUDY_SUFFIX=-mergesmoke ./phase3h-paper-study-mgsc-coarse-reltol.sh 2>&1 | tail -3

# --- 4) the 48-cell merge-levels sweep ---
for M in 1 2 3; do
  echo "=== MERGE=$M ==="
  MERGE=$M SC=false LEVELS="2 4 6 8" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3h-paper-study-mgsc-coarse-reltol.sh
  MERGE=$M SC=post  LEVELS="2 4 6 8" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3h-paper-study-mgsc-coarse-reltol.sh
  MERGE=$M SC=none  LEVELS="2 4 6 8" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3i-paper-study-localized-reltol.sh
  MERGE=$M SC=post  LEVELS="2 4 6 8" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3i-paper-study-localized-reltol.sh
done
echo "merge sweep done"
