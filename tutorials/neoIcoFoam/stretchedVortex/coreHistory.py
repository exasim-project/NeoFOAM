# SPDX-FileCopyrightText: 2023 - 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense
"""Track the vortex core radius and peak swirl over time.

Reads the ascii ``U`` files written by ``neoIcoFoam`` in this case, extracts the
mid-plane azimuthal velocity profile, and compares the measured core radius and
peak swirl against the exact unsteady Burgers solution at every write time, not
just against the final equilibrium.

The relaxation of a Gaussian-cored vortex under uniform axial strain has a
closed-form solution: with vorticity ``omega = (Gamma/pi delta^2) exp(-r^2/delta^2)``
the full Navier-Stokes system is satisfied exactly by

    delta(t)^2 = 4 nu / a + (delta_0^2 - 4 nu / a) exp(-a t)

so the core radius relaxes exponentially onto ``delta_eq = sqrt(4 nu / a)`` at
rate ``a``. That makes this a verification case with a known answer throughout
the transient, which is what lets the boundary treatment be checked rather than
assumed.

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

def read_mesh(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "system", "blockMeshDict")):
    """Cell counts and extents, read from blockMeshDict so setMeshDensity.py is
    the only place the resolution is written down."""
    text = open(path).read()
    hexa = re.search(r"hex \(0 1 2 3 4 5 6 7\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)", text)
    verts = re.findall(r"^\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\)", text, re.M)
    if not hexa or len(verts) < 8:
        sys.exit("could not read the mesh from " + path)
    counts = tuple(int(hexa.group(i)) for i in (1, 2, 3))
    pts = [tuple(float(v) for v in p) for p in verts]
    lo = tuple(min(p[i] for p in pts) for i in range(3))
    hi = tuple(max(p[i] for p in pts) for i in range(3))
    return counts, lo, hi


(NX, NY, NZ), (XMIN, YMIN, ZMIN), (XMAX, YMAX, ZMAX) = read_mesh()

NU = 0.01  # constant/transportProperties
A = 1.0  # strain rate
GAMMA = 1.0  # circulation
R0_INIT = 0.6  # initial core radius

# Burgers: u_theta = Gamma/(2 pi r) * (1 - exp(-r^2/delta^2))
DELTA_EQ = math.sqrt(4.0 * NU / A)


def delta_exact(t):
    """Exact core radius at time t (see module docstring)."""
    return math.sqrt(DELTA_EQ**2 + (R0_INIT**2 - DELTA_EQ**2) * math.exp(-A * t))
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

    def analytic(t):
        d = delta_exact(t)
        return S_PEAK * d, F_PEAK * GAMMA / (2.0 * math.pi * d)

    rows = []
    for t, name in times:
        r_num, u_num = peak(midplane_swirl(read_internal_vectors(os.path.join(name, "U"))))
        r_ex, u_ex = analytic(t)
        rows.append((t, r_num, u_num, r_ex, u_ex, 100.0 * (u_num - u_ex) / u_ex))

    with open("coreHistory.csv", "w") as fh:
        fh.write("time,r_peak,uTheta_peak,r_exact,uTheta_exact,error_percent\n")
        for row in rows:
            fh.write("{:g},{:.6g},{:.6g},{:.6g},{:.6g},{:.4g}\n".format(*row))

    r_eq, u_eq = S_PEAK * DELTA_EQ, F_PEAK * GAMMA / (2.0 * math.pi * DELTA_EQ)
    print(f"equilibrium: delta = {DELTA_EQ:.4f}  r_peak = {r_eq:.4f}  uTheta_peak = {u_eq:.4f}")
    print()
    print(f"{'time':>7}{'r_peak':>10}{'r_exact':>10}{'uTheta':>10}{'uTheta_ex':>11}{'err %':>9}")
    for t, r_num, u_num, r_ex, u_ex, err in rows:
        print(f"{t:>7g}{r_num:>10.4f}{r_ex:>10.4f}{u_num:>10.4f}{u_ex:>11.4f}{err:>9.1f}")
    print("\nWrote coreHistory.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ts = [r[0] for r in rows]
    fine = [ts[0] + i * (ts[-1] - ts[0]) / 200.0 for i in range(201)] if len(ts) > 1 else ts
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.plot(ts, [r[1] for r in rows], "o", label="computed")
    ax0.plot(fine, [analytic(t)[0] for t in fine], "-", c="k", label="exact")
    ax0.axhline(r_eq, ls=":", c="0.5", label="equilibrium")
    ax0.set_xlabel("t [s]")
    ax0.set_ylabel(r"core radius $r_{peak}$ [m]")
    ax0.legend()
    ax1.plot(ts, [r[2] for r in rows], "o", label="computed")
    ax1.plot(fine, [analytic(t)[1] for t in fine], "-", c="k", label="exact")
    ax1.axhline(u_eq, ls=":", c="0.5", label="equilibrium")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel(r"peak swirl $u_\theta$ [m/s]")
    ax1.legend()
    fig.tight_layout()
    fig.savefig("coreHistory.png", dpi=130)
    print("Wrote coreHistory.png")


if __name__ == "__main__":
    main()
