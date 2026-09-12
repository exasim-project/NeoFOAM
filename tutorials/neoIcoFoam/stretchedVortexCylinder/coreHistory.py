# SPDX-FileCopyrightText: 2023 - 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense
"""Track the vortex core radius and peak swirl, and compare with theory.

Same diagnostic as the box case, but this mesh is an O-grid, so cell positions
cannot be recovered from block counts: the cell centres are read from ``0/Cx``,
``0/Cy`` and ``0/Cz``, which ``Allrun`` writes with

    postProcess -func writeCellCentres -time 0

The strained vortex has a closed-form unsteady solution. With Gaussian vorticity
the core radius obeys ``d(delta^2)/dt = 4 nu - a delta^2``, hence

    delta(t)^2 = 4 nu / a + (delta_0^2 - 4 nu / a) exp(-a t)

so every write time has a known answer, not just the steady state.

Usage:
    python3 coreHistory.py            # writes coreHistory.csv (+ .png if matplotlib)
"""

import math
import os
import re
import sys

import numpy as np

NU = 0.01  # constant/transportProperties
A = 1.0  # strain rate
GAMMA = 1.0  # circulation
R0_INIT = 0.6  # initial core radius, setExprFieldsDict
DELTA_EQ = math.sqrt(4.0 * NU / A)
# max of (1 - exp(-s^2))/s occurs at s ~ 1.1209 with value ~ 0.63817
S_PEAK, F_PEAK = 1.12091, 0.63817
HERE = os.path.dirname(os.path.abspath(__file__))
VECTOR = re.compile(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)")


def read_internal(path, ncomp):
    text = open(path).read()
    head = text.index("internalField")
    if "nonuniform" not in text[head : head + 60]:
        sys.exit(f"{path}: expected a nonuniform internalField")
    start = text.index("(", text.index(">", head))
    body = text[start + 1 : text.index("\n)", start)]
    if ncomp == 3:
        return np.array([[float(c) for c in m.groups()] for m in VECTOR.finditer(body)])
    return np.fromstring(body, sep=" ")


def cell_centres():
    missing = [n for n in ("Cx", "Cy", "Cz") if not os.path.isfile(os.path.join(HERE, "0", n))]
    if missing:
        sys.exit("0/Cx..Cz missing -- run 'postProcess -func writeCellCentres -time 0' "
                 "(Allrun does it for you).")
    return tuple(read_internal(os.path.join(HERE, "0", n), 1) for n in ("Cx", "Cy", "Cz"))


def time_dirs():
    out = []
    for name in os.listdir(HERE):
        if os.path.isfile(os.path.join(HERE, name, "U")):
            try:
                out.append((float(name), name))
            except ValueError:
                pass
    return sorted(out)


def delta_exact(t):
    return math.sqrt(DELTA_EQ**2 + (R0_INIT**2 - DELTA_EQ**2) * math.exp(-A * t))


def main():
    cx, cy, cz = cell_centres()
    rad = np.hypot(cx, cy)
    levels = np.unique(np.round(cz, 9))
    midplane = np.isclose(cz, levels[np.argmin(np.abs(levels))])
    rmax, nbins = rad.max(), 60
    width = rmax / nbins
    bins = np.clip((rad[midplane] / width).astype(int), 0, nbins - 1)
    counts = np.bincount(bins, None, nbins)
    centres = (np.arange(nbins) + 0.5) * width

    times = time_dirs()
    if not times:
        sys.exit("No time directories with a U field found. Run ./Allrun first.")

    rows = []
    for t, name in times:
        U = read_internal(os.path.join(HERE, name, "U"), 3)
        swirl = np.where(rad > 1e-12, (-cy * U[:, 0] + cx * U[:, 1]) / np.maximum(rad, 1e-12), 0.0)
        totals = np.bincount(bins, swirl[midplane], nbins)
        profile = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
        k = int(np.nanargmax(profile))
        d = delta_exact(t)
        rows.append((t, centres[k], profile[k], S_PEAK * d, F_PEAK * GAMMA / (2 * math.pi * d)))

    with open(os.path.join(HERE, "coreHistory.csv"), "w") as fh:
        fh.write("time,r_peak,uTheta_peak,r_exact,uTheta_exact,error_percent\n")
        for t, r, u, rx, ux in rows:
            fh.write(f"{t:g},{r:.6g},{u:.6g},{rx:.6g},{ux:.6g},{(u - ux) / ux * 100:.3g}\n")

    print(f"equilibrium: delta = {DELTA_EQ:.4f}  r_peak = {S_PEAK * DELTA_EQ:.4f}  "
          f"uTheta_peak = {F_PEAK * GAMMA / (2 * math.pi * DELTA_EQ):.4f}")
    print(f"{'time':>7}{'r_peak':>10}{'r_exact':>10}{'uTheta':>10}{'uTheta_ex':>11}{'err %':>9}")
    for t, r, u, rx, ux in rows:
        print(f"{t:>7g}{r:>10.4f}{rx:>10.4f}{u:>10.4f}{ux:>11.4f}{(u - ux) / ux * 100:>9.1f}")
    print("\nWrote coreHistory.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ts = [r[0] for r in rows]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.plot(ts, [r[3] for r in rows], "-", c="0.25", lw=2, label=r"exact $1.121\,\delta(t)$")
    ax0.plot(ts, [r[1] for r in rows], "o", ms=4, label="measured")
    ax0.axhline(S_PEAK * DELTA_EQ, ls=":", c="0.5", label="equilibrium")
    ax0.set_xlabel("t [s]"); ax0.set_ylabel(r"core radius $r_{peak}$ [m]"); ax0.legend(fontsize=8)
    ax1.plot(ts, [r[4] for r in rows], "-", c="0.25", lw=2, label="exact")
    ax1.plot(ts, [r[2] for r in rows], "o", ms=4, label="measured")
    ax1.axhline(F_PEAK * GAMMA / (2 * math.pi * DELTA_EQ), ls=":", c="0.5", label="equilibrium")
    ax1.set_xlabel("t [s]"); ax1.set_ylabel(r"peak $u_\theta$ [m/s]"); ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "coreHistory.png"), dpi=130)
    print("Wrote coreHistory.png")


if __name__ == "__main__":
    main()
