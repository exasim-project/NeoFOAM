#!/usr/bin/env python3
"""Heatmaps of the param-study-mg-pmis.sh grid: ExecutionTime and pressure iters/step
vs (max_levels, PMIS strength_threshold). Crashed / incomplete cells are marked, since PMIS
coarsening is unstable at deep levels."""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, "paramStudyResults", "pmis-sweep")
OUT = os.path.join(HERE, "plots"); os.makedirs(OUT, exist_ok=True)

LEVELS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]                 # full sweep axes
THRS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
RUN = re.compile(r"pmis-L(\d+)-th(\d+p\d+)")

def status(log):
    """-> ('ok', exec_s, p_iters) | ('crash', ) | ('partial', steps) | None"""
    txt = open(log, errors="ignore").read().splitlines()
    steps = sum(1 for l in txt if l.startswith("Time = "))
    crashed = any(k in "\n".join(txt) for k in ("NotSupported", "terminate", "signal 8",
                                                "signal 6", "Aborted"))
    if steps >= 30:
        ex = [l for l in txt if "ExecutionTime" in l]
        if ex:
            try:
                et = float(ex[-1].split("ExecutionTime = ")[1].split(" s")[0])
                its = [int(l.split("No Iterations")[1].split()[0].rstrip(","))
                       for l in txt if "Solving for p," in l and "No Iterations" in l]
                return ("ok", et, np.mean(its) if its else np.nan)
            except (IndexError, ValueError):
                pass
    return ("crash",) if crashed else (("partial", steps) if steps else None)

# newest log per (L, thr)
grid = {}
for log in glob.glob(os.path.join(SW, "*.log")):
    m = RUN.search(os.path.basename(log))
    if not m:
        continue
    lev = int(m.group(1)); thr = float(m.group(2).replace("p", "."))
    mt = os.path.getmtime(log)
    if (lev, thr) not in grid or mt > grid[(lev, thr)][0]:
        grid[(lev, thr)] = (mt, log)

EX = np.full((len(THRS), len(LEVELS)), np.nan)
IT = np.full((len(THRS), len(LEVELS)), np.nan)
MARK = {}  # (i,j) -> 'X' crash / 'p' partial
nok = ncrash = 0
for (lev, thr), (_, log) in grid.items():
    if lev not in LEVELS or thr not in THRS:
        continue
    i, j = THRS.index(thr), LEVELS.index(lev)
    st = status(log)
    if st and st[0] == "ok":
        EX[i, j], IT[i, j] = st[1], st[2]; nok += 1
    elif st and st[0] == "crash":
        MARK[(i, j)] = "X"; ncrash += 1
    elif st and st[0] == "partial":
        MARK[(i, j)] = "p"

bi = np.unravel_index(np.nanargmin(EX), EX.shape)
print(f"completed: {nok}/90   crashed: {ncrash}")
print(f"fastest completed: L{LEVELS[bi[1]]} th{THRS[bi[0]]:g}  {EX[bi]:.1f}s  {IT[bi]:.0f} p-iters")
print("Pgm best-practice reference (localized MG, CG-10 coarse, cache): ~121s / 47 p-iters")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
for ax, M, title, cmap in [
    (axes[0], EX, "ExecutionTime [s]  (30 SIMPLE steps)", "viridis_r"),
    (axes[1], IT, "avg pressure iters / step", "magma_r"),
]:
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
    lo, hi = np.nanmin(M), np.nanmax(M)
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS)
    ax.set_yticks(range(len(THRS))); ax.set_yticklabels([f"{t:g}" for t in THRS])
    ax.set_xlabel("max_levels"); ax.set_ylabel("PMIS strength_threshold")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85)
    for i in range(len(THRS)):
        for j in range(len(LEVELS)):
            if not np.isnan(M[i, j]):
                c = "white" if (M[i, j]-lo)/(hi-lo+1e-9) > 0.55 else "black"
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7, color=c)
            elif MARK.get((i, j)) == "X":
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fc="0.85", ec="0.6", lw=0.4))
                ax.text(j, i, "✗", ha="center", va="center", fontsize=8, color="crimson")
            elif MARK.get((i, j)) == "p":
                ax.text(j, i, "·", ha="center", va="center", fontsize=10, color="0.5")
    ax.add_patch(plt.Rectangle((bi[1]-0.5, bi[0]-0.5), 1, 1, fill=False, ec="lime", lw=2.5))

fig.suptitle("PMIS coarsening sweep (localized MG, cache-rebuild100) — occDrivAer GPU H200   "
             f"[{nok}/90 completed, {ncrash} crashed ✗]\n"
             f"green = fastest completed: L{LEVELS[bi[1]]}, threshold {THRS[bi[0]]:g} → {EX[bi]:.0f}s / {IT[bi]:.0f} "
             f"p-iters   (vs Pgm best-practice ~121s / 47 iters)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
p = os.path.join(OUT, "9_pmis_sweep.png")
fig.savefig(p, dpi=140); plt.close(fig)
print("wrote", os.path.relpath(p, HERE))
