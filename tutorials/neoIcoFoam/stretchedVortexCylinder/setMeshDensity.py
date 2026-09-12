# SPDX-FileCopyrightText: 2023 - 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense
"""Scale the mesh density of this case.

The O-grid has three independent counts -- the core block is NI x NI, the four
outer blocks are NI x NR, and every block has NZ layers -- so all three are
scaled together and forced even, which keeps cell centres symmetric about the
axis and about the mid-plane.

Usage:

    python3 setMeshDensity.py 2          # twice the cells per direction
    python3 setMeshDensity.py 0.5        # half
    python3 setMeshDensity.py 2 --deltaT # also scale deltaT to hold the Courant number

Run it before ``./Allrun``. ``coreHistory.py`` reads the cell centres that
``Allrun`` writes, so it follows any density.
"""

import argparse
import os
import re
import sys

DELTA_EQ = 0.2  # equilibrium core radius
HERE = os.path.dirname(os.path.abspath(__file__))
BLOCKMESH = os.path.join(HERE, "system", "blockMeshDict")
CONTROLDICT = os.path.join(HERE, "system", "controlDict")
BLOCK = re.compile(r"(hex \([\d ]+\)\s*\()\s*(\d+)\s+(\d+)\s+(\d+)(\s*\))")
VERT = re.compile(r"^\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\)", re.M)


def even(n):
    return max(2, int(round(n / 2.0)) * 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("factor", type=float, help="cells per direction relative to the current mesh")
    ap.add_argument("--deltaT", action="store_true",
                    help="also scale controlDict deltaT, holding the Courant number")
    args = ap.parse_args()

    text = open(BLOCKMESH).read()
    blocks = BLOCK.findall(text)
    if len(blocks) != 5:
        sys.exit(f"expected the 5 O-grid blocks in {BLOCKMESH}, found {len(blocks)}")
    ni, nr, nz = int(blocks[1][1]), int(blocks[1][2]), int(blocks[1][3])
    verts = [tuple(float(v) for v in g) for g in VERT.findall(text)]
    zmax = max(abs(p[2]) for p in verts)
    radius = max(abs(p[0]) for p in verts) * 2 ** 0.5  # outer vertices sit at 45 degrees

    new = tuple(even(c * args.factor) for c in (ni, nr, nz))
    counts = [(new[0], new[0], new[2])] + [(new[0], new[1], new[2])] * 4

    out, idx = [], 0
    for line in text.splitlines(keepends=True):
        m = BLOCK.search(line)
        if m:
            c = counts[idx]; idx += 1
            line = line[: m.start()] + f"{m.group(1)}{c[0]} {c[1]} {c[2]}{m.group(5)}" \
                   + line[m.end():]
        out.append(line)
    open(BLOCKMESH, "w").write("".join(out))

    # the annulus spans radius - c*sqrt(2) in nr cells; the core block spans 2c in ni
    dz_old, dz_new = 2 * zmax / nz, 2 * zmax / new[2]
    cells_old = (ni * ni + 4 * ni * nr) * nz
    cells_new = (new[0] * new[0] + 4 * new[0] * new[1]) * new[2]
    print(f"mesh   NI {ni} -> {new[0]}, NR {nr} -> {new[1]}, NZ {nz} -> {new[2]}"
          f"  ({cells_new:,} cells, was {cells_old:,})")
    print(f"core   delta = {DELTA_EQ} is {DELTA_EQ / dz_new:.1f} cells axially "
          f"(was {DELTA_EQ / dz_old:.1f}); radius {radius:g}")

    ctext = open(CONTROLDICT).read()
    dt = float(re.search(r"^deltaT\s+([\d.eE+-]+);", ctext, re.M).group(1))
    usum = 1.0 * radius + 1.0 * zmax  # |u_r| + |u_z| at the rim, the Courant maximum
    co_now, co_new = usum * dt / dz_old, usum * dt / dz_new
    if args.deltaT:
        dt_new = dt * dz_new / dz_old
        ctext = re.sub(r"^deltaT\s+[\d.eE+-]+;", f"deltaT          {dt_new:g};", ctext, count=1,
                       flags=re.M)
        open(CONTROLDICT, "w").write(ctext)
        print(f"deltaT {dt:g} -> {dt_new:g}  (Courant held near {co_now:.2f})")
    else:
        print(f"Courant roughly {co_now:.2f} -> {co_new:.2f} at deltaT = {dt:g}"
              f"{'   -- rerun with --deltaT to hold it' if co_new > co_now else ''}")
    print(f"cost   roughly {cells_new / cells_old:.2f}x the current run per step")


if __name__ == "__main__":
    main()
