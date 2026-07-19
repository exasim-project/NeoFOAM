#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
# Resume ONLY the localized sc grid on the hoisted binary (global is already 36/36 complete).
# run_one skips cells whose logs exist, so the 7 finished cells are not redone. NO rebuild during.
SC=post LEVELS="2 3 4 5 6 8" TOLS="0.25 0.2 0.15 0.1 0.01 0.001" STEPS=50 \
  ./phase3i-paper-study-localized-reltol.sh
echo "localizedsc resume done"
