#!/usr/bin/env python3
"""Development of param-study-best.sh performance over time: ExecutionTime (and pressure
iters/step) of the best-practice runs vs wall-clock date, with optimisation milestones."""
import os, glob, re
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
BP = os.path.join(HERE, "paramStudyResults", "best-practice")
OUT = os.path.join(HERE, "plots"); os.makedirs(OUT, exist_ok=True)

SERIES = [  # (prefix, label, colour, marker)
    ("pMG-localized-best-cache-rebuild100",          "fp64 localized MG (best practice)", "#1f77b4", "o"),
    ("pMG-localized-precfloat-best-cache-rebuild100","float32 localized MG",              "#ff7f0e", "s"),
]
TS = re.compile(r"(\d{8}-\d{6})")

def parse(prefix):
    pts = []
    for log in glob.glob(os.path.join(BP, f"{prefix}-*.log")):
        try:
            txt = open(log, errors="ignore").read().splitlines()
        except OSError:
            continue
        if sum(1 for l in txt if l.startswith("Time = ")) != 30:   # complete 30-step runs only
            continue
        ex = [l for l in txt if "ExecutionTime" in l]
        if not ex:
            continue
        try:
            et = float(ex[-1].split("ExecutionTime = ")[1].split(" s")[0])
        except (IndexError, ValueError):
            continue
        m = TS.search(os.path.basename(log))
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
        its = [int(l.split("No Iterations")[1].split()[0].rstrip(","))
               for l in txt if "Solving for p," in l and "No Iterations" in l]
        pit = np.mean(its) if its else np.nan
        pts.append((t, et, pit))
    return sorted(pts)

data = {p: parse(p) for p, *_ in SERIES}

# milestones: (datetime, text)  -- placed at the transition where the gain first appears
MILES = [
    (datetime(2026, 6, 28, 10, 4),  "gradOpCtor fix\n(luw 51s→1.8s)"),
    (datetime(2026, 6, 29,  7, 30), "CG-10 coarse\nsolver"),
    (datetime(2026, 6, 30, 12, 0),  "CSR reuse +\nfloat/relTol defaults"),
]

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw=dict(height_ratios=[2.3, 1]))
for prefix, label, col, mk in SERIES:
    pts = data[prefix]
    if not pts:
        continue
    ts = [p[0] for p in pts]; et = [p[1] for p in pts]; pit = [p[2] for p in pts]
    ax.plot(ts, et, mk + "-", color=col, ms=6, lw=1.4, label=label, alpha=0.9)
    ax2.plot(ts, pit, mk + "-", color=col, ms=5, lw=1.2, alpha=0.9)
    # annotate best-practice fp64 first + last value
    if prefix.startswith("pMG-localized-best"):
        for t, e, _ in [pts[0], pts[-1]]:
            ax.annotate(f"{e:.0f}s", (t, e), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9, fontweight="bold", color=col)

for t, txt in MILES:
    for a in (ax, ax2):
        a.axvline(t, color="crimson", ls="--", lw=1.0, alpha=0.6)
    ax.text(t, 213, txt, rotation=0, fontsize=8.5,
            color="crimson", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="crimson", alpha=0.85))

ax.set_ylabel("ExecutionTime [s]  (30 SIMPLE steps)")
ax.set_title("param-study-best.sh — performance development over time (occDrivAer, GPU H200)",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3); ax.legend(loc="lower center", ncol=2, fontsize=9, framealpha=0.9)
ax.set_ylim(100, 222)
ax2.set_ylabel("avg pressure iters / step")
ax2.grid(True, alpha=0.3); ax2.set_xlabel("run date")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate(rotation=30)
fig.tight_layout()
p = os.path.join(OUT, "8_best_perf_timeline.png")
fig.savefig(p, dpi=140); plt.close(fig)
print("wrote", os.path.relpath(p, HERE))

# text summary: best-practice fp64 arc
bp = data["pMG-localized-best-cache-rebuild100"]
print(f"\nfp64 best-practice: first {bp[0][1]:.0f}s -> latest {bp[-1][1]:.0f}s  "
      f"({(1-bp[-1][1]/bp[0][1])*100:.0f}% faster), p-iters {bp[0][2]:.0f}->{bp[-1][2]:.0f}")
for prefix, label, *_ in SERIES:
    pts = data[prefix]
    if pts:
        print(f"  {label:34s} latest {pts[-1][1]:6.1f}s  ({pts[-1][0]:%m-%d %H:%M}, {len(pts)} runs)")
