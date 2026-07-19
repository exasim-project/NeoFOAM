#!/usr/bin/env python3
"""Overlay the force-coefficient development (Cd, Cl, Cs) of two production runs.

Built to compare the double-precision (fp64) MG production run against the earlier
float (float32) MG run, but takes any two runs. Re-run to refresh while a run is live.

Usage:
  python3 compare_force_runs.py                       # default fp64 vs float runs
  python3 compare_force_runs.py <runA> <runB>         # explicit run dirs (or .dat files)
"""
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

# (run-dir-or-datfile, label, colour)
DEFAULTS = [
    ("occDrivaerRun20260629-155942", "fp64 (double) MG", "#1f77b4"),
    ("occDrivaerRun20260629-100529", "float32 MG", "#ff7f0e"),
]


def dat_path(p):
    """Resolve a run dir (or direct .dat path) to its coefficient.dat."""
    if os.path.isfile(p):
        return p
    cand = os.path.join(p, "postProcessing", "neoForceCoeffs", "0", "coefficient.dat")
    if os.path.isfile(cand):
        return cand
    # any t0 subdir
    hits = glob.glob(os.path.join(p, "postProcessing", "neoForceCoeffs",
                                  "*", "coefficient.dat"))
    return hits[0] if hits else None


def load(p):
    f = dat_path(p if os.path.isabs(p) else os.path.join(ROOT, p))
    if not f or not os.path.isfile(f) or os.path.getsize(f) == 0:
        return None
    d = np.loadtxt(f, comments="#", ndmin=2)
    if d.size == 0:
        return None
    return d  # cols: 0 Time 1 Cd ... 4 Cl ... 10 Cs


def main(argv):
    if len(argv) >= 2:
        runs = [(argv[0], "run A", "#1f77b4"), (argv[1], "run B", "#ff7f0e")]
    else:
        runs = DEFAULTS

    series = [("Cd (drag)", 1), ("Cl (lift)", 4), ("Cs (side)", 10)]
    loaded = []
    for run, label, col in runs:
        d = load(run)
        if d is None:
            print(f"  [skip] no data for {run}")
            continue
        loaded.append((os.path.basename(run.rstrip("/")), label, col, d))
        print(f"  {label:18s} {os.path.basename(run.rstrip('/')):28s} "
              f"iters={len(d):5d}  Cd={d[-1,1]:.4f} Cl={d[-1,4]:.4f} Cs={d[-1,10]:.4f}")
    if not loaded:
        print("no runs with data"); return 1

    # overlap window = up to the shortest run's last iteration (the live run), so the
    # young run is not squashed by a long reference run. +5% margin.
    overlap = min(int(d[-1, 0]) for _, _, _, d in loaded)
    xmax = max(overlap * 1.05, 20)

    # quantify divergence between the first two runs over the common iterations
    if len(loaded) >= 2:
        da, db = loaded[0][3], loaded[1][3]
        n = min(len(da), len(db))
        print(f"  --- max |Δ| over first {n} common iters (fp64 - float) ---")
        for lab, ci in series:
            diff = np.abs(da[:n, ci] - db[:n, ci])
            print(f"    {lab:10s} max|Δ|={diff.max():.2e}  mean|Δ|={diff.mean():.2e}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, (lab, ci) in zip(axes, series):
        for run, label, col, d in loaded:
            ax.plot(d[:, 0], d[:, ci], color=col, lw=1.1, label=label,
                    alpha=0.85)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
    axes[0].set_xlim(0, xmax)
    axes[0].legend(loc="best", ncol=len(loaded))
    axes[-1].set_xlabel("SIMPLE iteration")

    desc = "  vs  ".join(f"{label} ({len(d)} it)" for _, label, _, d in loaded)
    fig.suptitle("Force-coefficient development — precision comparison\n"
                 + desc + f"\n(zoomed to overlap: 0–{overlap} iters)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(OUT, "6_force_compare_precision.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote:", os.path.relpath(p, HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
