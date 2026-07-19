#!/usr/bin/env python3
"""Plot the merge-levels sweep as heatmaps: merge{1,2,3} x max_levels{2,4,6,8}, one panel per config.

Data = 4 configs x max_levels{2,4,6,8} x merge{1,2,3} at coarse tol 0.1. merge2 is read from the
reltol grids (the -merge2 recheck was cancelled); merge1/merge3 from the -merge1/-merge3 dirs. Only
full 50-step cells are drawn; partials are shown as blank.

Output: merge-sweep-pms.png
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "paperParamStudyResults"
LEVELS = [2, 4, 6, 8]
MERGES = [(1, "-merge1"), (2, None), (3, "-merge3")]   # (label, dir suffix; None = reltol anchor)
CHAMP = 151  # corrected binding (0,1,2,3): global sc-post L6/tol0.1 (merge2). Pre-fix value was 722.

CONFIGS = [
    ("mgnosc-coarse-reltol",      "Global MG -- no scale correction"),
    ("mgscpost-coarse-reltol",    "Global MG -- sc post-pass"),
    ("localized-coarse-reltol",   "Localized MG -- no scale correction"),
    ("localizedsc-coarse-reltol", "Localized MG -- sc post-pass"),
]


def pms(study, L):
    g = sorted(glob.glob(f"{RESULTS}/{study}/L{L}e1-2026*.log"), reverse=True)
    if not g:
        return None
    lines = open(g[0], errors="ignore").readlines()
    if sum(1 for l in lines if l.startswith("Time = ")) < 50:
        return None
    v = [float(m.group(1)) for l in lines if "Solving for p," in l
         and (m := re.search(r"Solve time = ([0-9.]+)", l))]
    return sum(v) / len(v) if v else None


def matrix(study):
    M = np.full((len(MERGES), len(LEVELS)), np.nan)
    for i, (_, suffix) in enumerate(MERGES):
        src = study + suffix if suffix else study
        for j, L in enumerate(LEVELS):
            v = pms(src, L)
            if v is not None:
                M[i, j] = v
    return M


def main():
    mats = {s: matrix(s) for s, _ in CONFIGS}
    allv = [v for M in mats.values() for v in M.flatten() if not np.isnan(v)]
    # robust scale: cap at 90th pct so the merge1-shallow blow-ups (3000-5000) don't flatten the
    # 700-1100 band where the decisions live. Saturated cells still print their value.
    vmin, vmax = min(allv), float(np.percentile(allv, 90))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, (study, title) in zip(axes.ravel(), CONFIGS):
        M = mats[study].T   # transpose: rows = max_levels, cols = merge
        im = ax.imshow(M, cmap="viridis_r", aspect="auto", vmin=vmin, vmax=vmax)
        for i in range(len(LEVELS)):
            for j in range(len(MERGES)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                            fontsize=13, weight="bold",
                            color="white" if M[i, j] > (vmin + vmax) / 2 else "black")
        if not np.all(np.isnan(M)):
            bi, bj = np.unravel_index(np.nanargmin(M), M.shape)
            ax.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False,
                                       edgecolor="crimson", lw=2.5))
        ax.set_xticks(range(len(MERGES)))
        ax.set_xticklabels([f"merge {m}" for m, _ in MERGES], fontsize=12)
        ax.set_yticks(range(len(LEVELS)))
        ax.set_yticklabels([f"L{L}" for L in LEVELS], fontsize=12)
        ax.set_title(title, fontsize=13.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("coarsener", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("max_levels", fontsize=12)
    cb = fig.colorbar(im, ax=axes, label="mean pressure-solve time [ms]", shrink=0.85, pad=0.015, aspect=30)
    cb.set_label("mean pressure-solve time [ms]", fontsize=12)
    fig.suptitle(f"Merge-levels sweep: pressure-solve time (coarse tol 0.1; champion = {CHAMP} ms)",
                 fontsize=15)
    fig.savefig(f"{RESULTS}/merge-sweep-pms.png", dpi=140, bbox_inches="tight")
    print(f"wrote {RESULTS}/merge-sweep-pms.png")
    for study, title in CONFIGS:
        M = mats[study]
        if not np.all(np.isnan(M)):
            bi, bj = np.unravel_index(np.nanargmin(M), M.shape)
            print(f"  {title:38s} best {M[bi,bj]:.0f} ms  (merge{MERGES[bi][0]}, L{LEVELS[bj]})")


if __name__ == "__main__":
    main()
