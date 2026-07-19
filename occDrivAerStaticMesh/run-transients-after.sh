#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
# wait for the L2 row (which itself waits for the 35-cell sweep) so we don't contend for the GPUs
while ! grep -q "L2 row done" "$OUT/l2row.output" 2>/dev/null; do sleep 30; done
echo "=== L2 row finished; FORCE re-running the two transient cells ==="
# L5/1e-4 read 3.398 s/step / 23.1 iters -- neighbour L5/1e-3 is 2.702/14.3
# L6/0.1  read 3.244 s/step / 26.4 iters -- neighbours L6/0.15=2.681/17.1, L6/0.01=2.724/16.2
# Both are the startup-transient signature seen before (phase3f L3: 22.1 iters -> 11.2 on re-run).
FORCE=1 SC=false LEVELS="5" TOLS="0.0001" STEPS=50 ./phase3h-paper-study-mgsc-coarse-reltol.sh
FORCE=1 SC=false LEVELS="6" TOLS="0.1"    STEPS=50 ./phase3h-paper-study-mgsc-coarse-reltol.sh
echo "transients done"
