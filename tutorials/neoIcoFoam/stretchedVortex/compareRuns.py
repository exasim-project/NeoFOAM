# SPDX-FileCopyrightText: 2023 - 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense
"""Compare several stretchedVortex runs against the exact unsteady solution.

Put any number of finished runs side by side -- square box against cylinder,
one convection scheme against another, one mesh density against another -- and
see what each one does to the amplification the case is about.

    python3 compareRuns.py box=../stretchedVortex cyl=../stretchedVortexCylinder
    python3 compareRuns.py upwind=run_upwind linear=run_linear -o schemes.png

Each argument is ``label=path``. A run is read from its reconstructed time
directories; the mesh is taken from ``0/Cx,Cy,Cz`` when they exist (any mesh,
written by ``postProcess -func writeCellCentres``) and otherwise from a
single-block ``blockMeshDict``, so box runs work without extra steps.

The panels are:

1. peak ``u_theta`` against the exact ``0.638 Gamma / (2 pi delta(t))``;
2. amplification ``Gamma / (pi delta^2)`` -- the peak vorticity of a Gaussian
   core -- on a log axis, against the exact curve and against the inviscid
   ``exp(a t)`` that the same vortex would follow with ``nu = 0``. This is the
   "blowup" panel: it shows how far each run tracks the growth before viscosity
   arrests it, and how much of the difference between runs is numerical;
3. relative error in peak ``u_theta``;
4. centreline ``u_z / (a z)``, which the exact solution holds at 1.

``delta`` is inferred from the measured core radius, ``delta = r_peak / 1.121``,
so panel 2 is a restatement of the measured profile rather than an independent
vorticity measurement -- no gradients are taken, which is what lets it work on
any mesh.
"""

import argparse
import math
import os
import re
import sys

import numpy as np

NU, A, GAMMA, R0 = 0.01, 1.0, 1.0, 0.6
DELTA_EQ = math.sqrt(4.0 * NU / A)
S_PEAK, F_PEAK = 1.12091, 0.63817
AXIS_R = 0.08  # cells this close to the axis count as "centreline"
VECTOR = re.compile(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)")


def read_internal(path, ncomp):
    text = open(path).read()
    head = text.index("internalField")
    if "nonuniform" not in text[head : head + 60]:
        raise RuntimeError(f"{path}: uniform internalField")
    start = text.index("(", text.index(">", head))
    body = text[start + 1 : text.index("\n)", start)]
    if ncomp == 3:
        return np.array([[float(c) for c in m.groups()] for m in VECTOR.finditer(body)])
    return np.fromstring(body, sep=" ")


def cell_centres(case):
    """Cell centres, from writeCellCentres output or from a single-block mesh."""
    zero = os.path.join(case, "0")
    if all(os.path.isfile(os.path.join(zero, n)) for n in ("Cx", "Cy", "Cz")):
        return tuple(read_internal(os.path.join(zero, n), 1) for n in ("Cx", "Cy", "Cz"))
    dict_ = os.path.join(case, "system", "blockMeshDict")
    text = open(dict_).read()
    hexa = re.search(r"hex \(0 1 2 3 4 5 6 7\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)", text)
    if not hexa:
        sys.exit(f"{case}: no 0/Cx..Cz and no single-block mesh -- run "
                 "'postProcess -func writeCellCentres -time 0' in that case")
    n = [int(hexa.group(i)) for i in (1, 2, 3)]
    verts = re.findall(r"^\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\)", text, re.M)
    pts = [tuple(float(v) for v in p) for p in verts]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    ax = [lo[i] + (np.arange(n[i]) + 0.5) * (hi[i] - lo[i]) / n[i] for i in range(3)]
    # blockMesh numbers cells with i fastest, then j, then k
    cz, cy, cx = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
    return cx.ravel(), cy.ravel(), cz.ravel()


def history(case):
    cx, cy, cz = cell_centres(case)
    rad = np.hypot(cx, cy)
    levels = np.unique(np.round(cz, 9))
    midplane = np.isclose(cz, levels[np.argmin(np.abs(levels))])
    top = np.isclose(cz, levels[-1]) & (rad < AXIS_R)
    nbins, width = 60, rad.max() / 60
    bins = np.clip((rad[midplane] / width).astype(int), 0, nbins - 1)
    counts = np.bincount(bins, None, nbins)
    centres = (np.arange(nbins) + 0.5) * width

    rows = []
    for name in sorted(os.listdir(case)):
        if not os.path.isfile(os.path.join(case, name, "U")):
            continue
        try:
            t = float(name)
        except ValueError:
            continue
        U = read_internal(os.path.join(case, name, "U"), 3)
        if len(U) != len(rad):
            sys.exit(f"{case}/{name}: {len(U)} cells but the mesh has {len(rad)}")
        swirl = np.where(rad > 1e-12, (-cy * U[:, 0] + cx * U[:, 1]) / np.maximum(rad, 1e-12), 0.0)
        totals = np.bincount(bins, swirl[midplane], nbins)
        profile = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
        k = int(np.nanargmax(profile))
        rows.append((t, centres[k], profile[k], U[top, 2].mean() / (A * levels[-1])))
    if not rows:
        sys.exit(f"{case}: no reconstructed time directories with a U field")
    return np.array(sorted(rows))



def blowup_figure(runs, out, plt):
    """How close does each run stay to unarrested growth?

    The core amplification is Gamma/(pi delta^2); with nu = 0 the same vortex
    would grow as exp(a t) for ever. Two readings of "close to blowup":

      retained fraction  A(t) / A(0) exp(a t)   -- 1 means still on the
                                                   inviscid trajectory
      growth exponent    d ln A / dt / a        -- 1 inviscid, 0 fully arrested

    delta is taken from the measured peak swirl, delta = 0.638 Gamma / (2 pi u),
    which is smooth, rather than from the radially binned core radius.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = plt.get_cmap("tab10")

    tt = np.linspace(0, max(r[-1, 0] for _, r in runs), 400)
    d_ex = np.sqrt(DELTA_EQ**2 + (R0**2 - DELTA_EQ**2) * np.exp(-A * tt))
    amp_ex = GAMMA / (np.pi * d_ex**2)
    ax[0].semilogy(tt, amp_ex / (amp_ex[0] * np.exp(A * tt)), "-", c="0.25", lw=2.4,
                   label="exact")
    g_ex = np.gradient(np.log(amp_ex), tt) / A
    ax[1].plot(tt, g_ex, "-", c="0.25", lw=2.4, label="exact")

    print(f"{'run':<16}{'retained at t=6':>17}{'t where dlnA/dt < a/2':>24}")
    for i, (label, r) in enumerate(runs):
        c = colors(i % 10)
        t = r[:, 0]
        delta_u = F_PEAK * GAMMA / (2 * np.pi * r[:, 2])
        amp = GAMMA / (np.pi * delta_u**2)
        retained = amp / (amp[0] * np.exp(A * t))
        g = np.gradient(np.log(amp), t) / A
        ax[0].semilogy(t, retained, "o", ms=3.2, color=c, label=label)
        ax[1].plot(t, g, "-", lw=1.8, color=c, label=label)
        below = t[g < 0.5]
        t_half = below[0] if len(below) else float("nan")
        print(f"{label:<16}{retained[-1]:>17.4f}{t_half:>24.2f}")

    ax[0].set_ylabel(r"$A(t)\,/\,A(0)e^{at}$")
    ax[0].set_title("fraction of the inviscid growth retained")
    ax[1].axhline(1.0, ls="--", c="0.55", lw=1.4)
    ax[1].axhline(0.0, ls=":", c="0.55", lw=1.2)
    ax[1].set_ylim(-0.15, 1.15)
    ax[1].set_ylabel(r"$(d\ln A/dt)\,/\,a$")
    ax[1].set_title("growth rate: 1 = inviscid, 0 = arrested")
    for a_ in ax:
        a_.set_xlabel("t [s]"); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="label=path")
    ap.add_argument("-o", "--out", default="methodComparison.png")
    ap.add_argument("--blowup", action="store_true",
                    help="instead plot how much of the unarrested inviscid growth each run keeps")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for spec in args.runs:
        if "=" not in spec:
            ap.error(f"expected label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        runs.append((label, history(path)))

    if args.blowup:
        blowup_figure(runs, args.out, plt)
        return

    tmax = max(r[-1, 0] for _, r in runs)
    tt = np.linspace(0, tmax, 300)
    delta = np.sqrt(DELTA_EQ**2 + (R0**2 - DELTA_EQ**2) * np.exp(-A * tt))
    u_ex = F_PEAK * GAMMA / (2 * np.pi * delta)
    amp_ex = GAMMA / (np.pi * delta**2)
    amp_inviscid = GAMMA / (np.pi * R0**2) * np.exp(A * tt)
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.4))
    ax[0, 0].plot(tt, u_ex, "-", c="0.25", lw=2.4, label="exact")
    ax[0, 1].semilogy(tt, amp_ex, "-", c="0.25", lw=2.4, label=r"exact $\Gamma/\pi\delta(t)^2$")
    ax[0, 1].semilogy(tt, amp_inviscid, "--", c="0.55", lw=1.8, label=r"inviscid $\propto e^{at}$")
    ax[1, 1].axhline(1.0, ls=":", c="0.25", lw=1.4)

    print(f"{'run':<14}{'peak err %':>12}{'mean err %':>12}{'centreline':>12}{'times':>7}")
    for i, (label, r) in enumerate(runs):
        c = colors(i % 10)
        d = np.sqrt(DELTA_EQ**2 + (R0**2 - DELTA_EQ**2) * np.exp(-A * r[:, 0]))
        ux = F_PEAK * GAMMA / (2 * np.pi * d)
        amp = GAMMA / (np.pi * (r[:, 1] / S_PEAK) ** 2)
        err = (r[:, 2] - ux) / ux * 100
        ax[0, 0].plot(r[:, 0], r[:, 2], "o", ms=3.2, color=c, label=label)
        ax[0, 1].semilogy(r[:, 0], amp, "o", ms=3.2, color=c, label=label)
        ax[1, 0].plot(r[:, 0], err, "-", lw=1.8, color=c, label=label)
        ax[1, 1].plot(r[:, 0], r[:, 3], "-", lw=1.8, color=c, label=label)
        print(f"{label:<14}{np.abs(err).max():>12.2f}{np.abs(err).mean():>12.2f}"
              f"{r[-1, 3]:>12.3f}{len(r):>7}")

    ax[0, 0].set_ylabel(r"peak $u_\theta$ [m/s]"); ax[0, 0].set_title("swirl amplification")
    ax[0, 1].set_ylabel(r"$\Gamma/\pi\delta^2$  (peak vorticity of the core)")
    ax[0, 1].set_title("growth, against the unarrested inviscid rate")
    ax[1, 0].set_ylabel(r"peak $u_\theta$ error [%]"); ax[1, 0].set_title("error vs the exact solution")
    ax[1, 1].set_ylabel(r"centreline $u_z / (a z)$")
    ax[1, 1].set_title("is the strain reaching the axis?")
    for a_ in ax.ravel():
        a_.set_xlabel("t [s]"); a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
