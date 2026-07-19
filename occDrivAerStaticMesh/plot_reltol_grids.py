#!/usr/bin/env python3
"""Plot the max_levels x coarse-rel-tol grids for all four preconditioner variants.

Panels: global MG with scale correction (both passes) | without | post-smooth pass only |
        localized Schwarz{MG}.

All timings are STEADY STATE, (ET_last - ET_first)/(steps-1): the naive ExecutionTime/steps folds in
~47 s of one-time setup and inflates a 50-step window by ~0.9 s/step. Only full 50-step cells are
plotted; partial cells are omitted rather than shown misleadingly.

Output: reltol-grids-{time,iters}.png
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "paperParamStudyResults"
TOL_CODE = {0.25: "025", 0.2: "02", 0.15: "015",
            0.1: "1", 0.01: "2", 0.001: "3", 0.0001: "4"}
# Plotted tolerances. 1e-4 is measured (see peek-grid.py) but omitted here: it is the worst cell in
# every row of every panel -- over-solving the coarse system buys no outer iterations and costs up to
# +27 % wall time -- so it only widens the figure. Cells outside this list are dropped at LOAD time so
# the colour scale, the per-panel optimum and the plotted cells all derive from the same set.
TOLS = [0.25, 0.2, 0.15, 0.1, 0.01, 0.001]
# Plotted max_levels. L10 exists in the localized no-sc grid (legacy {4,6,8,10} range) but is dropped
# here so all four panels share the same {2..8} depth axis; its data stays in the logs and peek-grid.
PLOT_LEVELS = [2, 3, 4, 5, 6, 8]

PANELS = [
    # 2x2: rows = architecture (global / localized), cols = no sc / sc post-pass.
    # sc grids are on the HOISTED + faithful port (D^-1 inner op, fused dots, no per-call cudaMalloc).
    # no-sc grids are port-independent (sc=off never enters the scale blocks).
    ("mgnosc-coarse-reltol",     "Global MG -- no scale correction"),
    ("mgscpost-coarse-reltol",   "Global MG -- sc post-pass"),
    ("localized-coarse-reltol",  "Localized MG -- no scale correction"),
    ("localizedsc-coarse-reltol","Localized MG -- sc post-pass"),
]


def cell(path):
    """(steady_s_per_step, mean_outer_iters, mean_p_solve_ms, steps) for a full window, else None."""
    try:
        lines = open(path, errors="ignore").readlines()
    except OSError:
        return None
    steps = sum(1 for l in lines if l.startswith("Time = "))
    et = [float(m.group(1)) for l in lines if (m := re.search(r"ExecutionTime = ([0-9.]+) s", l))]
    if steps < 50 or len(et) < 2:
        return None                       # partial windows are excluded, never plotted
    steady = (et[-1] - et[0]) / (steps - 1)
    its = [int(m.group(1)) for l in lines if "Solving for p," in l
           and (m := re.search(r"No Iterations (\d+)", l))]
    pms = [float(m.group(1)) for l in lines if "Solving for p," in l
           and (m := re.search(r"Solve time = ([0-9.]+)", l))]
    if not its:
        return None
    return steady, float(np.mean(its)), float(np.mean(pms)), steps


def load(study):
    """{(levels, tol): (steady, iters, p_ms)}"""
    d = os.path.join(RESULTS, study)
    out = {}
    if not os.path.isdir(d):
        return out
    for f in glob.glob(f"{d}/L*e*-2026*.log"):
        m = re.match(r"L(\d+)e(\d+)$", os.path.basename(f).split("-2026")[0])
        if not m:
            continue
        lvl, code = int(m.group(1)), m.group(2)
        if lvl not in PLOT_LEVELS:
            continue                      # drop L10 etc. from scale, best-cell, and grid alike
        tol = next((t for t, c in TOL_CODE.items() if c == code), None)
        if tol is None or tol not in TOLS:
            continue                      # unplotted tolerances must not skew the colour scale
        c = cell(f)
        if c:
            # keep the newest full run if a cell was re-run (transient FORCE re-runs)
            key = (lvl, tol)
            if key not in out or f > out[key][1]:
                out[key] = (c, f)
    return {k: v[0] for k, v in out.items()}


def draw(metric, idx, label, fname, fmt="{:.2f}"):
    data = {s: load(s) for s, _ in PANELS}
    vals = [v[idx] for d in data.values() for v in d.values()]
    if not vals:
        print("no data at all")
        return
    # Robust shared colour scale. A single breakdown cell (the sc/L4/0.25 cliff, 3.90 s) would
    # otherwise stretch the range and flatten the 2.4-3.0 band where every meaningful difference
    # lives. Clip the top at the 90th percentile and let outliers saturate; their value is still
    # printed in the cell, so nothing is hidden -- only the colour is capped.
    vmin, vmax = min(vals), float(np.percentile(vals, 90))
    if vmax <= vmin:
        vmax = max(vals)
    fig, axes2d = plt.subplots(2, 2, figsize=(13.5, 8.6), constrained_layout=True)
    axes = axes2d.ravel()
    for ax, (study, title) in zip(axes, PANELS):
        d = data[study]
        if not d:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    fontsize=13, color="0.5", transform=ax.transAxes)
            ax.set_title(title, fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        levels = [L for L in sorted({k[0] for k in d}) if L in PLOT_LEVELS]
        M = np.full((len(levels), len(TOLS)), np.nan)
        for i, L in enumerate(levels):
            for j, t in enumerate(TOLS):
                if (L, t) in d:
                    M[i, j] = d[(L, t)][idx]
        im = ax.imshow(M, cmap="viridis_r", aspect="auto", vmin=vmin, vmax=vmax)
        for i in range(len(levels)):
            for j in range(len(TOLS)):
                if not np.isnan(M[i, j]):
                    best = M[i, j] == np.nanmin(M)
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                            fontsize=9, weight="bold" if best else "normal",
                            color="white" if M[i, j] > (vmin + vmax) / 2 else "black")
        # ring the global optimum of this panel
        if not np.all(np.isnan(M)):
            bi, bj = np.unravel_index(np.nanargmin(M), M.shape)
            ax.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False,
                                       edgecolor="red", lw=2.2))
        ax.set_xticks(range(len(TOLS)))
        ax.set_xticklabels([f"{t:g}" for t in TOLS], rotation=45, fontsize=12)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([f"L{L}" for L in levels], fontsize=12)
        ax.set_title(title, fontsize=14)
    # 2x2: label only the outer edges, so the inner panels stay uncluttered
    for ax in axes2d[-1, :]:
        ax.set_xlabel("coarse relative tolerance", fontsize=13)
    for ax in axes2d[:, 0]:
        ax.set_ylabel("max_levels", fontsize=13)
    cb=fig.colorbar(im, ax=axes2d, label=label, shrink=0.85, pad=0.015, aspect=30)
    cb.set_label(label, fontsize=13); cb.ax.tick_params(labelsize=11)
    fig.suptitle(f"Pressure preconditioner: max_levels x coarse rel-tol -- {label}", fontsize=16)
    fig.savefig(f"{RESULTS}/{fname}", dpi=140, bbox_inches="tight")
    print(f"wrote {RESULTS}/{fname}")
    for study, title in PANELS:
        d = data[study]
        if d:
            k = min(d, key=lambda k: d[k][idx])
            print(f"  {study:26s} best L{k[0]} tol={k[1]:<7g} "
                  f"{d[k][0]:.3f} s/step  {d[k][1]:.1f} iters  {d[k][2]:.0f} ms")
        else:
            print(f"  {study:26s} (no full-window cells yet)")


if __name__ == "__main__":
    # p_ms is the primary metric: it is what these sweeps actually change (momentum/turbulence/assemble
    # are identical across cells, so s/step dilutes the pressure-solve signal ~10x).
    draw("pms", 2, "mean pressure-solve time [ms]", "reltol-grids-pms.png", "{:.0f}")
    draw("time", 0, "steady-state wall time per SIMPLE iteration [s]", "reltol-grids-time.png", "{:.2f}")
    draw("iters", 1, "mean pressure (outer CG) iterations per solve", "reltol-grids-iters.png", "{:.0f}")
