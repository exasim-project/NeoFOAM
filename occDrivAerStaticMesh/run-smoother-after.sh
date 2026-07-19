#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
# chain behind the localized MG grid -- one job on the 4 GPUs at a time
while ! grep -q "localized grid done" "$OUT/localized.output" 2>/dev/null; do sleep 30; done
echo "=== localized grid finished; STEP 1: is smoothing local? ==="
STEPS=50 ./phase3j-paper-study-localized-smoother.sh step1
echo "=== STEP 2: localized Chebyshev degree x levels ==="
STEPS=50 ./phase3j-paper-study-localized-smoother.sh step2
echo "localized-smoother all done"
