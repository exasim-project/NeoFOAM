# SPDX-FileCopyrightText: 2023 - 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense
"""Scale the mesh density of this case.

The mesh has to stay uniform and symmetric: the prescribed boundary flux only
balances to round-off because the strain part is constant on each face and the
swirl part cancels between cell centres either side of the axis (see
``system/setExprBoundaryFieldsDict``). ``neoIcoFoam`` does not call ``adjustPhi``,
so a mesh that breaks that symmetry leaves the pressure equation with an
inconsistent right-hand side. This script therefore only ever changes the cell
counts, keeps ``dx = dy = dz``, and forces every count even.

Usage:

    python3 setMeshDensity.py 2          # twice the cells per direction
    python3 setMeshDensity.py 0.5        # half
    python3 setMeshDensity.py --dx 0.02  # target spacing instead of a factor
    python3 setMeshDensity.py 2 --deltaT # also scale deltaT to hold the Courant number

Run it before ``./Allrun``; ``blockMesh`` picks the new counts up. ``coreHistory.py``
reads the counts back out of ``blockMeshDict``, so it follows automatically.
"""

import argparse
import math
import os
import re
import sys

BASE_DX = 0.03125  # the counts below are the reference mesh
DELTA_EQ = 0.2  # equilibrium core radius, sqrt(4 nu / a)
HERE = os.path.dirname(os.path.abspath(__file__))
BLOCKMESH = os.path.join(HERE, "system", "blockMeshDict")
CONTROLDICT = os.path.join(HERE, "system", "controlDict")
HEX = re.compile(r"(hex \(0 1 2 3 4 5 6 7\)\s*\()\s*(\d+)\s+(\d+)\s+(\d+)(\s*\))")
VERT = re.compile(r"^\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\)", re.M)


def read_geometry():
    text = open(BLOCKMESH).read()
    m = HEX.search(text)
    if not m:
        sys.exit("could not find the block definition in system/blockMeshDict")
    counts = tuple(int(m.group(i)) for i in (2, 3, 4))
    verts = [tuple(float(v) for v in g) for g in VERT.findall(text)]
    if len(verts) < 8:
        sys.exit("could not read the vertices from system/blockMeshDict")
    lo = tuple(min(v[i] for v in verts) for i in range(3))
    hi = tuple(max(v[i] for v in verts) for i in range(3))
    return text, counts, lo, hi


def even(n):
    """Cell counts must be even so cell centres stay symmetric about the axis."""
    return max(2, int(round(n / 2.0)) * 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("factor", nargs="?", type=float, default=None,
                    help="cells per direction relative to the current mesh")
    ap.add_argument("--dx", type=float, default=None, help="target cell size instead")
    ap.add_argument("--deltaT", action="store_true",
                    help="also scale controlDict deltaT, holding the Courant number")
    args = ap.parse_args()
    if (args.factor is None) == (args.dx is None):
        ap.error("give either a factor or --dx")

    text, counts, lo, hi = read_geometry()
    span = tuple(hi[i] - lo[i] for i in range(3))
    dx_now = span[0] / counts[0]

    factor = args.factor if args.factor is not None else dx_now / args.dx
    new = tuple(even(c * factor) for c in counts)
    dx_new = tuple(span[i] / new[i] for i in range(3))
    if max(dx_new) - min(dx_new) > 1e-9:
        print(f"note: rounding to even counts made the cells slightly anisotropic "
              f"({dx_new[0]:.5g}, {dx_new[1]:.5g}, {dx_new[2]:.5g})")

    text = HEX.sub(lambda m: f"{m.group(1)}{new[0]} {new[1]} {new[2]}{m.group(5)}", text, count=1)
    text = re.sub(r"(// Box .*uniform dx = dy = dz = )[\d.]*\d",
                  lambda m: f"{m.group(1)}{dx_new[0]:g}", text, count=1)
    open(BLOCKMESH, "w").write(text)

    cells = new[0] * new[1] * new[2]
    print(f"mesh   {counts[0]} x {counts[1]} x {counts[2]} -> {new[0]} x {new[1]} x {new[2]}"
          f"  ({cells:,} cells, dx = {dx_new[0]:g})")
    print(f"core   delta = {DELTA_EQ} is {DELTA_EQ / dx_new[0]:.1f} cells "
          f"(was {DELTA_EQ / dx_now:.1f})")

    ctext = open(CONTROLDICT).read()
    dt = float(re.search(r"^deltaT\s+([\d.eE+-]+);", ctext, re.M).group(1))
    # OpenFOAM's Courant number sums the face fluxes, so it goes with
    # |u_x| + |u_y| + |u_z|, not |U|. The maximum sits at a top corner, where the
    # strain gives a(|x| + |y|)/2 in plane (the swirl only shuffles it between the
    # two components) and a|z| axially.
    a = 1.0
    usum = a * (abs(lo[0]) + abs(lo[1])) / 2.0 + a * max(abs(lo[2]), abs(hi[2]))
    co_now = usum * dt / dx_now
    co_new = usum * dt / dx_new[0]
    if args.deltaT:
        dt_new = dt * dx_new[0] / dx_now
        ctext = re.sub(r"^deltaT\s+[\d.eE+-]+;", f"deltaT          {dt_new:g};", ctext, count=1,
                       flags=re.M)
        open(CONTROLDICT, "w").write(ctext)
        print(f"deltaT {dt:g} -> {dt_new:g}  (Courant held at {co_now:.2f})")
    else:
        print(f"Courant {co_now:.2f} -> {co_new:.2f} at deltaT = {dt:g}"
              f"{'   -- rerun with --deltaT to hold it' if co_new > co_now else ''}")
    print(f"cost   roughly {cells / 196608:.2f}x the reference 64 x 64 x 48 run per step")


if __name__ == "__main__":
    main()
