#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
while ! grep -q "scpost grid done" "$OUT/scpost.output" 2>/dev/null; do sleep 30; done
echo "=== scpost finished; starting the LOCALIZED grid ==="
LEVELS="4 6 8 10" TOLS="0.25 0.2 0.15 0.1 0.01 0.001 0.0001" STEPS=50 \
  ./phase3i-paper-study-localized-reltol.sh
echo "localized grid done"
