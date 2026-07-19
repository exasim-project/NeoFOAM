#!/usr/bin/env python3
"""Figure 2.4 for PaperOptimizationStudy: per-SIMPLE-iteration wall time (normalized
to the first iteration) and pressure/momentum linear-solver iteration counts over the
0->1000 spin-up window. Reads the Phase-0 spin-up log directly.

Usage: python3 plot_spinup_timing.py [spinup.log] [out.png]
"""
import re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else \
    "paperParamStudyResults/spinup/spinup-1000it-20260709-111720.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "paperParamStudyResults/spinup-timing-iterations.png"

steps, et, p, u = [], [], [], []
cur = None
def flush():
    global cur
    if cur and cur["et"] is not None:
        steps.append(cur["t"]); et.append(cur["et"]); p.append(cur["p"]); u.append(cur["u"])
    cur = None
with open(LOG) as f:
    for line in f:
        m = re.match(r"Time = (\d+)\s*$", line)
        if m:
            flush(); cur = {"t": int(m.group(1)), "p": None, "u": None, "et": None}; continue
        if cur is None: continue
        m = re.search(r"Solving for Ux.*No Iterations (\d+)", line)
        if m: cur["u"] = int(m.group(1))
        m = re.search(r"Solving for p,.*No Iterations (\d+)", line)
        if m: cur["p"] = int(m.group(1))
        m = re.search(r"ExecutionTime = ([\d.]+) s", line)
        if m: cur["et"] = float(m.group(1))
flush()

# marginal per-step wall time (delta ExecutionTime), defined from step 2 onward;
# step 1's ExecutionTime folds in the one-time setup, so normalize to the first marginal.
m_steps = steps[1:]
m_dt = [et[i] - et[i-1] for i in range(1, len(et))]
t0 = m_dt[0]
norm = [d / t0 for d in m_dt]
plateau = sum(m_dt[200:]) / len(m_dt[200:]) / t0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True)
ax1.plot(m_steps, norm, color="#1f77b4", lw=0.9)
ax1.axhline(1.0, color="0.6", lw=0.8, ls=":")
ax1.axhline(plateau, color="0.35", lw=0.9, ls="--",
            label=f"developed plateau ~ {plateau:.2f} (~ {plateau*t0:.2f} s/step)")
ax1.set_ylabel("per-iteration solve time\n(normalized to first iteration)")
ax1.set_title("SIMPLE spin-up 0->1000: per-iteration wall time and solver iteration counts")
ax1.legend(loc="upper right", fontsize=9, frameon=False); ax1.grid(True, alpha=0.25)

ax2.plot(steps, p, color="#d62728", lw=0.8, label="pressure  (Cg + Multigrid)")
ax2.plot(steps, u, color="#2ca02c", lw=0.8, label="momentum  (Schwarz(Jacobi)+BiCGStab, Ux)")
ax2.set_xlabel("SIMPLE iteration"); ax2.set_ylabel("linear-solver iterations")
ax2.legend(loc="upper right", fontsize=9, frameon=False); ax2.grid(True, alpha=0.25)
ax2.set_xlim(0, steps[-1])

fig.tight_layout(); fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT, f"(t0={t0:.2f}s, plateau={plateau*t0:.2f}s/step)")
