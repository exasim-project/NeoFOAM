#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
while ! grep -q "d2 rerun done" "$OUT/d2rerun.output" 2>/dev/null; do sleep 20; done
echo "=== d2 rerun finished; localized MG x scale correction ==="
STEPS=50 ./phase3l-localized-sc.sh
