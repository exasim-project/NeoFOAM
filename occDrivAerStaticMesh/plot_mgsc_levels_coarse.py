#!/usr/bin/env python3
"""Heatmap of marginal s/step over the max_levels x coarse_iters grid, for the global MG with
MG-level scale correction + pgmMerge3 (p-multigrid-mgsc-m3-L*-c*). Sequential blue ramp (light=fast,
dark=slow); each cell annotated s/step (p-iters). Reads the run logs directly."""
import matplotlib, glob, os, re
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

LS = [2, 3, 4, 5, 6, 8, 10]
CS = [4, 6, 8, 10, 16]
RESULTS = "paperParamStudyResults/mgsc-levels-coarse"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

def _steps(f):
    t = open(f).read()
    return t.count("\nTime = ") + t.startswith("Time = ")

def cell_metrics(L, C):
    logs = glob.glob(f"{RESULTS}/L{L}c{C}-2026*.log")
    if not logs:
        return None, None
    # prefer the most-complete run (max steps); tie-break on newest mtime -- so an in-progress
    # partial re-run is ignored until it reaches the full step count, then it wins.
    f = max(logs, key=lambda p: (_steps(p), os.path.getmtime(p)))
    txt = open(f).read()
    steps = txt.count("\nTime = ") + txt.startswith("Time = ")
    ets = re.findall(r"ExecutionTime = ([0-9.]+) s", txt)
    its = [int(m) for m in re.findall(r"Solving for p,.*?No Iterations (\d+)", txt)]
    if not ets or steps == 0:
        return None, None
    sstep = float(ets[-1]) / steps
    pit = sum(its) / len(its) if its else None
    return sstep, pit

grid = np.full((len(LS), len(CS)), np.nan)
pit = np.full((len(LS), len(CS)), np.nan)
for i, L in enumerate(LS):
    for j, C in enumerate(CS):
        s, p = cell_metrics(L, C)
        if s is not None:
            grid[i, j] = s
            pit[i, j] = p if p is not None else np.nan

# sequential blue ramp (palette steps 100->650), light = fast (good), dark = slow
blues = ["#e8f1fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
cmap = mcolors.LinearSegmentedColormap.from_list("seqblue", blues)
vmin, vmax = np.nanmin(grid), np.nanmin([np.nanmax(grid), 5.0])  # clamp so one hot cell doesn't wash out

fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

# annotate each cell: s/step (p-iters); text ink flips on dark cells
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
best = np.nanmin(grid)
for i in range(len(LS)):
    for j in range(len(CS)):
        if np.isnan(grid[i, j]):
            ax.text(j, i, "—", ha="center", va="center", color=INK2, fontsize=10); continue
        dark = norm(min(grid[i, j], vmax)) > 0.55
        tc = "#ffffff" if dark else INK
        star = "★\n" if abs(grid[i, j] - best) < 1e-6 else ""
        ax.text(j, i, f"{star}{grid[i,j]:.2f}\n({pit[i,j]:.0f})", ha="center", va="center",
                color=tc, fontsize=8.6, linespacing=1.15,
                fontweight="bold" if star else "normal")

ax.set_xticks(range(len(CS))); ax.set_xticklabels(CS)
ax.set_yticks(range(len(LS))); ax.set_yticklabels(LS)
ax.set_xlabel("coarsest-solver iterations", fontsize=10.5, color=INK2)
ax.set_ylabel("max_levels", fontsize=10.5, color=INK2)
ax.set_title("MG-level sc + pgmMerge3: s/step (p-iters) over levels × coarse iters",
             fontsize=11.5, color=INK, pad=10, loc="left")
ax.tick_params(colors=INK2, labelsize=10)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, len(CS), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(LS), 1), minor=True)
ax.grid(which="minor", color=SURFACE, lw=2.5); ax.tick_params(which="minor", length=0)

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("marginal s/step  (lighter = faster)", fontsize=9, color=INK2)
cb.ax.tick_params(colors=INK2, labelsize=8); cb.outline.set_visible(False)

fig.text(0.01, 0.008, "occDrivAer 4×H200, 50 steps, cached. ★ = fastest cell. Optimum basin L4–L5 × "
         "lean coarse (c4–c6); too shallow (L2) or c16 over-solve both cost more.",
         fontsize=6.8, color=INK2, ha="left")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = "paperParamStudyResults/mgsc-levels-coarse-heatmap.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out, "| best %.2f" % best)
