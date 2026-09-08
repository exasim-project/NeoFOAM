# SPDX-FileCopyrightText: 2023 - 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense
"""Track the vortex core radius and peak swirl over time.

Reads the ascii ``U`` files written by ``neoIcoFoam`` in this case, extracts the
mid-plane azimuthal velocity profile, and compares the measured core radius and
peak swirl against the analytic Burgers equilibrium.

Assumes the single-block uniform mesh from ``system/blockMeshDict``: blockMesh
numbers cells with i fastest, then j, then k, so the cell centre of index
``n = i + NX*(j + NY*k)`` is known analytically and no mesh parsing is needed.

Usage:
    python coreHistory.py            # writes coreHistory.csv (+ .png if matplotlib)
"""

import math
import os
import re
import sys

# Must match system/blockMeshDict and system/setExprFieldsDict.
NX, NY, NZ = 64, 64, 32
XMIN, XMAX = -1.0, 1.0
YMIN, YMAX = -1.0, 1.0
ZMIN, ZMAX = -0.5, 0.5

NU = 0.01  # constant/transportProperties
A = 1.0  # strain rate
GAMMA = 1.0  # circulation
R0_INIT = 0.6  # initial core radius

# Burgers equilibrium: u_theta = Gamma/(2 pi r) * (1 - exp(-r^2/delta^2))
DELTA_EQ = math.sqrt(4.0 * NU / A)
# max of (1 - exp(-s^2))/s occurs at s ~ 1.1209 with value ~ 0.63817
S_PEAK, F_PEAK = 1.12091, 0.63817

VECTOR = re.compile(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)")


def time_dirs(root="."):
    out = []
    for name in os.listdir(root):
        if not os.path.isfile(os.path.join(root, name, "U")):
            continue
        try:
            out.append((float(name), name))
        except ValueError:
            pass
    return sorted(out)


def read_internal_vectors(path):
    """Return the internalField of an ascii volVectorField as a list of tuples."""
    with open(path) as fh:
        text = fh.read()
    head = text.index("internalField")
    # NB: "nonuniform" contains "uniform", so test for it first.
    if "nonuniform" not in text[head : head + 60]:
        value = VECTOR.search(text, head).groups()
        return [tuple(float(c) for c in value)] * (NX * NY * NZ)
    open_paren = text.index("(", text.index("List<vector>", head))
    close_paren = text.index("\n)", open_paren)
    body = text[open_paren + 1 : close_paren]
    return [tuple(float(c) for c in m.groups()) for m in VECTOR.finditer(body)]


def midplane_swirl(field):
    """Azimuthal velocity magnitude vs radius on the k = NZ/2 cell layer."""
    dx, dy = (XMAX - XMIN) / NX, (YMAX - YMIN) / NY
    k = NZ // 2
    samples = []
    for j in range(NY):
        y = YMIN + (j + 0.5) * dy
        for i in range(NX):
            x = XMIN + (i + 0.5) * dx
            r = math.hypot(x, y)
            if r < 1.0e-9:
                continue
            ux, uy, _ = field[i + NX * (j + NY * k)]
            samples.append((r, (-y * ux + x * uy) / r))
    samples.sort()
    return samples


def peak(samples, nbins=60, rmax=1.0):
    """Radially bin the profile and return (r_peak, u_theta_peak)."""
    width = rmax / nbins
    total = [0.0] * nbins
    count = [0] * nbins
    for r, ut in samples:
        b = int(r / width)
        if b < nbins:
            total[b] += ut
            count[b] += 1
    best_r, best_u = 0.0, -1.0e30
    for b in range(nbins):
        if count[b] == 0:
            continue
        mean = total[b] / count[b]
        if mean > best_u:
            best_u, best_r = mean, (b + 0.5) * width
    return best_r, best_u


def main():
    times = time_dirs()
    if not times:
        sys.exit("No time directories with a U field found. Run ./Allrun first.")

    u_eq = F_PEAK * GAMMA / (2.0 * math.pi * DELTA_EQ)
    r_eq = S_PEAK * DELTA_EQ
    u_init = F_PEAK * GAMMA / (2.0 * math.pi * R0_INIT)
    r_init = S_PEAK * R0_INIT

    rows = []
    for t, name in times:
        r_peak, u_peak = peak(midplane_swirl(read_internal_vectors(os.path.join(name, "U"))))
        rows.append((t, r_peak, u_peak))

    with open("coreHistory.csv", "w") as fh:
        fh.write("time,r_peak,uTheta_peak\n")
        for t, r_peak, u_peak in rows:
            fh.write(f"{t:g},{r_peak:.6g},{u_peak:.6g}\n")

    print(f"analytic initial   : r_peak = {r_init:.4f}  uTheta_peak = {u_init:.4f}")
    print(f"analytic equilibrium: r_peak = {r_eq:.4f}  uTheta_peak = {u_eq:.4f}")
    print()
    print(f"{'time':>8}{'r_peak':>12}{'uTheta_peak':>14}{'u/u_init':>12}")
    for t, r_peak, u_peak in rows:
        print(f"{t:>8g}{r_peak:>12.4f}{u_peak:>14.4f}{u_peak / u_init:>12.2f}")
    print("\nWrote coreHistory.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ts = [r[0] for r in rows]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.plot(ts, [r[1] for r in rows], "o-", label="measured")
    ax0.axhline(r_eq, ls="--", c="k", label="Burgers equilibrium")
    ax0.set_xlabel("t [s]")
    ax0.set_ylabel(r"core radius $r_{peak}$ [m]")
    ax0.legend()
    ax1.plot(ts, [r[2] for r in rows], "o-", label="measured")
    ax1.axhline(u_eq, ls="--", c="k", label="Burgers equilibrium")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel(r"peak swirl $u_\theta$ [m/s]")
    ax1.legend()
    fig.tight_layout()
    fig.savefig("coreHistory.png", dpi=130)
    print("Wrote coreHistory.png")


if __name__ == "__main__":
    main()
