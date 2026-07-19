#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
# merge3 pass: 4 configs x max_levels{2,4,6,8} x merge3 at coarse tol 0.1 = 16 cells.
# Env-var prefixes must be LITERAL tokens, so each config is invoked explicitly (not via an
# expanded "$scarg" -- bash treats an expanded VAR=val as a command name, not an assignment).
export MERGE=3
export STUDY_SUFFIX="-merge3"
export STEPS=50
LV="2 4 6 8"
TL="0.1"

echo "=== merge3: global no-sc ==="
SC=false LEVELS="$LV" TOLS="$TL" ./phase3h-paper-study-mgsc-coarse-reltol.sh

echo "=== merge3: global sc-post ==="
SC=post LEVELS="$LV" TOLS="$TL" ./phase3h-paper-study-mgsc-coarse-reltol.sh

echo "=== merge3: localized no-sc ==="
SC=none LEVELS="$LV" TOLS="$TL" ./phase3i-paper-study-localized-reltol.sh

echo "=== merge3: localized sc-post ==="
SC=post LEVELS="$LV" TOLS="$TL" ./phase3i-paper-study-localized-reltol.sh

echo "merge3 done"
