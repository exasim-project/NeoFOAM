#!/bin/bash
cd /storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh
OUT=/scratch/greole/tmp/claude-1039/-storage-home-greole-code-NeoFOAM/53ce7522-35f6-42ec-8923-d10545f303c1/tasks
while ! grep -q "sc grids done" "$OUT/scgrids2.output" 2>/dev/null; do sleep 30; done
echo "=== sweeps complete; detecting + re-running anomalies ==="

rerun_grid() {   # $1 = study, $2 = SC, $3 = driver script
    local study="$1" sc="$2" drv="$3"
    echo "--- $study ---"
    ./rerun-anomalies.py "$study" > /tmp/anom-$study.txt 2>&1
    cat /tmp/anom-$study.txt
    grep -q "no anomalies" /tmp/anom-$study.txt && return
    # capture the cells BEFORE deleting, so we can report the before/after
    ./rerun-anomalies.py "$study" --delete > /tmp/anom-$study-del.txt 2>&1
    local lv tl
    lv=$(grep -oP 'LEVELS="\K[^"]+' /tmp/anom-$study-del.txt)
    tl=$(grep -oP 'TOLS="\K[^"]+'   /tmp/anom-$study-del.txt)
    [ -z "$lv" ] && return
    echo ">> re-running LEVELS=\"$lv\" TOLS=\"$tl\"  (logs deleted, so run_one will redo exactly these)"
    SC="$sc" LEVELS="$lv" TOLS="$tl" STEPS=50 "./$drv"
}

rerun_grid mgscpost-coarse-reltol     post phase3h-paper-study-mgsc-coarse-reltol.sh
rerun_grid localizedsc-coarse-reltol  post phase3i-paper-study-localized-reltol.sh

echo "=== post-rerun check (should report none) ==="
./rerun-anomalies.py mgscpost-coarse-reltol
./rerun-anomalies.py localizedsc-coarse-reltol
echo "anomaly rerun done"
