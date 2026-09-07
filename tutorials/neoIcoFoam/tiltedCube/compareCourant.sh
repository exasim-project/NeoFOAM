#!/bin/bash
# Run tiltedCube serial vs parallel (CPU executor) and diff the per-step max Courant
# number. Serial is the oracle; a correct distributed solve must match it.
#
#   ./compareCourant.sh [NPROCS]
#
# Honors NEOFOAM_BIN (default develop build). Uses UCX_TLS=tcp for WSL2 MPI.
#
# NOTE: deliberately does NOT use `set -u` or source OpenFOAM RunFunctions --
# RunFunctions references unbound shell vars and trips `set -u`.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
NEOFOAM_BIN="${NEOFOAM_BIN:-$HERE/../../../build/develop/bin/neoIcoFoam}"
NPROCS="${1:-4}"

if [ ! -d constant/polyMesh ]; then
    echo "No mesh found -- running ./Allmesh first"
    ./Allmesh || { echo "Allmesh failed"; exit 1; }
fi

echo "=== SERIAL (binary: $NEOFOAM_BIN) ==="
rm -rf 0 processor*
cp -r 0.orig 0
"$NEOFOAM_BIN" > log.serial 2>&1
echo "serial exit=$?"

echo "=== PARALLEL (np=$NPROCS, scotch, executor CPU) ==="
decomposePar -force > log.decomposePar 2>&1
grep -E 'Number of processor faces|Number of cells =' log.decomposePar | sed 's/^/  /'
UCX_TLS=tcp mpirun -np "$NPROCS" "$NEOFOAM_BIN" -parallel > log.parallel 2>&1
echo "parallel exit=$?"

echo
echo "  step | serial maxCo            | parallel maxCo           | absdiff"
paste \
  <(grep -E 'Courant Number' log.serial   | sed -E 's/.*max: //') \
  <(grep -E 'Courant Number' log.parallel | awk '!seen[$0]++' | sed -E 's/.*max: //') \
| awk '{d=$1-$2; if(d<0)d=-d; printf "  %4d | %-22s | %-22s | %.3e\n", NR, $1, $2, d}'

echo
echo "blowup / FPE markers:"
grep -iE 'FOAM FATAL|signal 8|nan|inf|bounding' log.serial log.parallel 2>/dev/null | head -8
echo "(end)"
