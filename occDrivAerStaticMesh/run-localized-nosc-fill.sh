#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
# wait for the localized SC sweep to finish so we don't contend for the 4 GPUs
while ! grep -q "localizedsc resume done" "$OUT/locscresume.output" 2>/dev/null; do sleep 30; done
echo "=== localized SC done; filling localized NO-SC grid to {2,3,4,5,6,8} ==="
# SC=none, full level range. run_one SKIPS L4/L6/L8 (already present) and fills L2/L3/L5 = 18 new cells.
# no-sc is port-independent (sc=off never enters the scale blocks), so these are comparable to the
# existing L4/L6/L8 rows despite the binary changes since -- but note the hoisted binary is identical
# for the no-sc path anyway.
SC=none LEVELS="2 3 4 5 6 8" TOLS="0.25 0.2 0.15 0.1 0.01 0.001" STEPS=50 \
  ./phase3i-paper-study-localized-reltol.sh
echo "localized nosc fill done"
