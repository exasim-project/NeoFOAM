#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
while ! grep -q "sc grids done" "$OUT/scgrids2.output" 2>/dev/null; do sleep 20; done
echo "=== sweeps done. Quantify the alloc-hoist so we know whether the mixed-binary sweep matters ==="
# Re-run the SAME cell (global L4/0.1) that the pre-hoist binary measured at p_ms=814 / 9.3 iters.
# The hoist removes ~37 synchronizing raw cudaMalloc/cudaFree pairs per solve (eps + relax scalars).
#   - iters MUST be identical (9.3): the hoist changes allocation, not math.
#   - p_ms delta = the true cost of the alloc churn. If <1% (the noise floor), the 69 pre-hoist cells
#     are still comparable and the sweep stands. If larger, both grids need re-running on one binary.
rm -f paperParamStudyResults/mgscpost-coarse-reltol/L4e1-2026*.log
SC=post LEVELS="4" TOLS="0.1" STEPS=50 ./phase3h-paper-study-mgsc-coarse-reltol.sh
echo "hoist check done"
