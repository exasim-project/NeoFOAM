#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
# chain behind the transient re-runs (which chain behind the L2 row) -- one job on the 4 GPUs at a time
while ! grep -q "transients done" "$OUT/transients.output" 2>/dev/null; do sleep 30; done
echo "=== transients finished; starting the SC=post 2-D grid ==="
SC=post LEVELS="2 3 4 5 6" TOLS="0.25 0.2 0.15 0.1 0.01 0.001 0.0001" STEPS=50 \
  ./phase3h-paper-study-mgsc-coarse-reltol.sh
echo "scpost grid done"
