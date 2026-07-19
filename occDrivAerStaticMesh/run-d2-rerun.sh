#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
while ! grep -q "chebyshev-foci done" "$OUT/stepB.output" 2>/dev/null; do sleep 20; done
echo "=== stepB finished; FORCE re-running the d2 outlier ==="
# d2 read 31.1 iters against d1=16.8 and d3=15.6 -- non-monotone, so almost certainly a transient
# (a real degree ceiling would degrade monotonically). Same signature as the 4+ transients already
# confirmed this session. FORCE=1 because run_one skips when a log exists.
FORCE=1 FOCI_HI=2.0 STEPS=50 ./phase3k-paper-study-chebyshev-foci.sh stepB
echo "d2 rerun done"
