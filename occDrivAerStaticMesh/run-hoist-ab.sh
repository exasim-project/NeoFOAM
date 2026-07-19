#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
# B: same cell, hoisted binary. The hoist removes ~37 synchronizing raw cudaMalloc/cudaFree pairs per
# solve (dzc's `eps` + the relaxation scalar, both now persistent workspace slots 11/12).
#   iters MUST be 9.3 -- the hoist is not a numerical change.
#   p_ms delta = the true cost of the alloc churn.
#     <1%  -> noise; the 69 pre-hoist cells stay comparable and both grids stand.
#     >>1% -> the mixed-binary sweep is invalid and both grids need re-running on one binary.
# Two repeats, because the expected effect (~0.2% by my estimate) is at/below the noise floor and a
# single run cannot resolve it -- and my cost estimates have been badly wrong before (the Allreduce
# model was off 8x).
for i in 1 2; do
  rm -f paperParamStudyResults/mgscpost-coarse-reltol/L4e1-2026*.log
  SC=post LEVELS="4" TOLS="0.1" STEPS=50 ./phase3h-paper-study-mgsc-coarse-reltol.sh
  cp paperParamStudyResults/mgscpost-coarse-reltol/L4e1-2026*.log \
     paperParamStudyResults/_hoist-ab/B-hoisted-run$i.log 2>/dev/null
done
echo "hoist ab done"
