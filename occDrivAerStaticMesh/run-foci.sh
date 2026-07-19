#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
STEPS=50 ./phase3k-paper-study-chebyshev-foci.sh stepA
echo "=== stepA done; now the queued localized-sc test ==="
STEPS=50 ./run-locsc-inline.sh 2>/dev/null || true
echo "foci sweep done"
