#!/usr/bin/env python3
"""Heatmaps of the param-study-mg-level-sweep.sh grid: ExecutionTime and pressure iters/step
vs (max_levels, coarse-CG max_iters). Crashed / incomplete cells are marked. The localized float
MG preconditioner (precfloat base) with a fixed-iteration CG coarsest solver, cache-rebuild100."""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, "paramStudyResults", "level-sweep")
OUT = os.path.join(HERE, "plots"); os.makedirs(OUT, exist_ok=True)

LEVELS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]                  # full sweep axes
CITERS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]              # coarse-CG max_iters
RUN = re.compile(r"precfloat-L(\d+)-coarsecg(\d+)")

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

# newest log per (L, citers)
grid = {}
for log in glob.glob(os.path.join(SW, "*.log")):
    m = RUN.search(os.path.basename(log))
    if not m:
        continue
    lev = int(m.group(1)); cit = int(m.group(2))
    mt = os.path.getmtime(log)
    if (lev, cit) not in grid or mt > grid[(lev, cit)][0]:
        grid[(lev, cit)] = (mt, log)

NTOT = len(LEVELS) * len(CITERS)
EX = np.full((len(CITERS), len(LEVELS)), np.nan)
IT = np.full((len(CITERS), len(LEVELS)), np.nan)
MARK = {}  # (i,j) -> 'X' crash / 'p' partial
nok = ncrash = 0
for (lev, cit), (_, log) in grid.items():
    if lev not in LEVELS or cit not in CITERS:
        continue
    i, j = CITERS.index(cit), LEVELS.index(lev)
    st = status(log)
    if st and st[0] == "ok":
        EX[i, j], IT[i, j] = st[1], st[2]; nok += 1
    elif st and st[0] == "crash":
        MARK[(i, j)] = "X"; ncrash += 1
    elif st and st[0] == "partial":
        MARK[(i, j)] = "p"

if nok:
    bi = np.unravel_index(np.nanargmin(EX), EX.shape)
    print(f"completed: {nok}/{NTOT}   crashed: {ncrash}")
    print(f"fastest completed: L{LEVELS[bi[1]]} coarsecg{CITERS[bi[0]]}  "
          f"{EX[bi]:.1f}s  {IT[bi]:.0f} p-iters")
else:
    bi = None
    print(f"completed: 0/{NTOT}   crashed: {ncrash} -- no completed runs to rank")
print("Pgm best-practice reference (localized MG, CG-10 coarse, cache): ~121s / 47 p-iters")

fig, axes = plt.subplots(1, 2, figsize=(15, 7.0))
for ax, M, title, cmap in [
    (axes[0], EX, "ExecutionTime [s]  (30 SIMPLE steps)", "viridis_r"),
    (axes[1], IT, "avg pressure iters / step", "magma_r"),
]:
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
    lo, hi = (np.nanmin(M), np.nanmax(M)) if np.isfinite(M).any() else (0.0, 1.0)
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS)
    ax.set_yticks(range(len(CITERS))); ax.set_yticklabels(CITERS)
    ax.set_xlabel("max_levels"); ax.set_ylabel("coarse-CG max_iters")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85)
    for i in range(len(CITERS)):
        for j in range(len(LEVELS)):
            if not np.isnan(M[i, j]):
                c = "white" if (M[i, j]-lo)/(hi-lo+1e-9) > 0.55 else "black"
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7, color=c)
            elif MARK.get((i, j)) == "X":
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fc="0.85", ec="0.6", lw=0.4))
                ax.text(j, i, "✗", ha="center", va="center", fontsize=8, color="crimson")
            elif MARK.get((i, j)) == "p":
                ax.text(j, i, "·", ha="center", va="center", fontsize=10, color="0.5")
    if bi is not None:
        ax.add_patch(plt.Rectangle((bi[1]-0.5, bi[0]-0.5), 1, 1, fill=False, ec="lime", lw=2.5))

best = (f"green = fastest completed: L{LEVELS[bi[1]]}, coarse-CG {CITERS[bi[0]]} iters "
        f"→ {EX[bi]:.0f}s / {IT[bi]:.0f} p-iters   (vs Pgm best-practice ~121s / 47 iters)"
        if bi is not None else "no completed runs yet")
fig.suptitle("max_levels × coarse-CG-iters sweep (localized float MG, cache-rebuild100) — "
             f"occDrivAer GPU H200   [{nok}/{NTOT} completed, {ncrash} crashed ✗]\n" + best,
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
p = os.path.join(OUT, "10_level_sweep.png")
fig.savefig(p, dpi=140); plt.close(fig)
print("wrote", os.path.relpath(p, HERE))
