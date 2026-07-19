#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
# Validate the restored relaxation factor BEFORE committing to the 72-cell sweep.
#   L2/0.25 : collapsed to 122.4 iters undamped. Archived unfaithful-port value: 18.3.
#   L4/0.1  : was UNaffected (9.4). Must stay ~9.4 -- proves the fix didn't break what worked.
SC=post LEVELS="2" TOLS="0.25" STEPS=50 ./phase3h-paper-study-mgsc-coarse-reltol.sh
SC=post LEVELS="4" TOLS="0.1"  STEPS=50 ./phase3h-paper-study-mgsc-coarse-reltol.sh
echo "relax check done"
