#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
# wait for the 35-cell no-sc sweep to finish so we don't contend for the 4 GPUs
while ! grep -q "phase3h NOSC done" "$OUT/b0lvj39x5.output" 2>/dev/null; do sleep 30; done
echo "=== no-sc sweep finished; starting L2 row ==="
SC=false LEVELS="2" TOLS="0.25 0.2 0.15 0.1 0.01 0.001 0.0001" STEPS=50 \
  ./phase3h-paper-study-mgsc-coarse-reltol.sh
echo "L2 row done"
