#!/bin/bash
#
# Re-run, under the CORRECTED one-rank-per-GPU binding (paper-study-common.sh now exports
# CUDA_VISIBLE_DEVICES=0,1,2,3), the studies whose ABSOLUTE timings the oversubscription bug inflated.
# Every run is FORCE=1 so a fresh, post-fix, timestamped log is written; the plot scripts pick the
# newest log per cell, so the corrected numbers win automatically.
#
# Order (user's priority; front-loads the short ones, huge rel-tol grid in the middle):
#   1. phase1  cost-breakdown  (reference Cg+MG, 3 instrumented runs)
#   2. phase3  cache-compare   (2x2 cache/rebuild, 250 steps each)
#   3. rel-tol grids           (4 panels x levels{2,3,4,5,6,8} x tols{.25..001} = 144 cells)  [merge2 anchor]
#   4. merge-sweep             (merge1 + merge3 only; merge2 == the rel-tol grid) x 4 configs x L{2,4,6,8}
#
# Each stage is an independent process with its own dictionary backup/restore trap. Progress lands in
# paperParamStudyResults/<study>/ exactly where the plotters read it.

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
export FORCE=1

banner() { echo; echo "############################################################"; echo "# $*"; echo "############################################################"; }

# grid the rel-tol figure actually plots (plot_reltol_grids.py: PLOT_LEVELS, TOLS)
RTLEVELS="2 3 4 5 6 8"
RTTOLS="0.25 0.2 0.15 0.1 0.01 0.001"
MERGE_LEVELS="2 4 6 8"   # plot_merge_sweep.py LEVELS

t0=$SECONDS

banner "STAGE 1/4  phase1 cost-breakdown  (corrected binding)"
./phase1-paper-study-costbreakdown.sh

banner "STAGE 2/4  phase3 cache-compare  (corrected binding)"
./phase3-paper-study-cache-compare.sh

banner "STAGE 3/4  rel-tol grids  (144 cells; merge2 anchor)"
SC=false LEVELS="$RTLEVELS" TOLS="$RTTOLS" STEPS=50 ./phase3h-paper-study-mgsc-coarse-reltol.sh   # Global MG, no sc
SC=post  LEVELS="$RTLEVELS" TOLS="$RTTOLS" STEPS=50 ./phase3h-paper-study-mgsc-coarse-reltol.sh   # Global MG, sc post-pass
SC=none  LEVELS="$RTLEVELS" TOLS="$RTTOLS" STEPS=50 ./phase3i-paper-study-localized-reltol.sh     # Localized, no sc
SC=post  LEVELS="$RTLEVELS" TOLS="$RTTOLS" STEPS=50 ./phase3i-paper-study-localized-reltol.sh     # Localized, sc post-pass

banner "STAGE 4/4  merge-sweep  (merge1 + merge3)"
for M in 1 3; do
    banner "  merge $M"
    MERGE=$M SC=false LEVELS="$MERGE_LEVELS" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3h-paper-study-mgsc-coarse-reltol.sh
    MERGE=$M SC=post  LEVELS="$MERGE_LEVELS" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3h-paper-study-mgsc-coarse-reltol.sh
    MERGE=$M SC=none  LEVELS="$MERGE_LEVELS" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3i-paper-study-localized-reltol.sh
    MERGE=$M SC=post  LEVELS="$MERGE_LEVELS" TOLS="0.1" STEPS=50 STUDY_SUFFIX="-merge$M" ./phase3i-paper-study-localized-reltol.sh
done

banner "ALL STAGES DONE  (wall $(( (SECONDS - t0) / 60 )) min)"
echo "Re-plot: python3 plot_reltol_grids.py ; python3 plot_merge_sweep.py"
