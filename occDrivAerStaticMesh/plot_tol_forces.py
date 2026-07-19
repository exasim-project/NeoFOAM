#!/usr/bin/env python3
"""Plot Cd/Cl/Cs and runtimes for the param-study-production-tol.sh pressure-relTol sweep."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNROOT = "/storage/home/greole/code/NeoFOAM"
CASE = "/storage/home/greole/code/NeoFOAM/occDrivAerStaticMesh"
# relTol -> (ExecutionTime s, ClockTime s, avg p-iters/step)  [from the run logs]
RUNTIME = {
    "0.01": (2071.29, 2086, 21.9),
    "0.02": (1939.82, 1954, 16.5),
    "0.03": (1872.95, 1888, 13.5),
}
TOLS = ["0.01", "0.02", "0.03"]              # the 3 complete (1000-step) runs
REF = {"Cd": 0.26, "Cl": 0.00121, "Cs": 0.0}  # reference values
COL = {"Cd": 1, "Cl": 4, "Cs": 10}            # column indices in coefficient.dat
colors = {"0.01": "#1f77b4", "0.02": "#ff7f0e", "0.03": "#2ca02c"}

def load(tol):
    f = glob.glob(f"{RUNROOT}/occDrivaerRun*-relTol{tol}/postProcessing/neoForceCoeffs/0/coefficient.dat")
    d = np.loadtxt(f[0], comments="#")
    return d

data = {t: load(t) for t in TOLS}

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("occDrivAer production: pressure-relTol sweep (1000 SIMPLE steps, fp64 localized MG best-practice)",
             fontsize=13, fontweight="bold")

# --- force time series (Cd, Cl, Cs) ---
for k, key in enumerate(["Cd", "Cl", "Cs"]):
    a = ax.flat[k]
    for t in TOLS:
        d = data[t]
        a.plot(d[:, 0], d[:, COL[key]], color=colors[t], lw=0.8, alpha=0.85,
               label=f"relTol={t}")
        # converged mean (last 300 steps)
        m = d[d[:, 0] >= d[-1, 0] - 300][:, COL[key]].mean()
        a.axhline(m, color=colors[t], ls=":", lw=1.0, alpha=0.7)
    a.axhline(REF[key], color="k", ls="--", lw=1.2, label=f"ref={REF[key]:g}")
    a.set_title(f"{key}  (dotted = last-300-step mean)")
    a.set_xlabel("SIMPLE step"); a.set_ylabel(key)
    a.grid(alpha=0.3); a.legend(fontsize=8, ncol=2)
ax.flat[1].set_ylim(-0.05, 0.20)   # zoom Cl around its small range

# --- runtime + p-iters comparison ---
a = ax.flat[3]
xs = np.arange(len(TOLS))
exec_t = [RUNTIME[t][0] for t in TOLS]
piters = [RUNTIME[t][2] for t in TOLS]
b = a.bar(xs - 0.18, exec_t, 0.36, color="#4c72b0", label="ExecutionTime [s]")
a.bar_label(b, fmt="%.0f", fontsize=9)
a.set_xticks(xs); a.set_xticklabels([f"relTol\n{t}" for t in TOLS])
a.set_ylabel("ExecutionTime [s]", color="#4c72b0")
a.set_ylim(0, max(exec_t) * 1.18)
a.set_title("Runtime vs pressure relTol (1000 steps)")
a2 = a.twinx()
b2 = a2.bar(xs + 0.18, piters, 0.36, color="#dd8452", label="avg p-iters/step")
a2.bar_label(b2, fmt="%.1f", fontsize=9)
a2.set_ylabel("avg pressure iters / step", color="#dd8452")
a2.set_ylim(0, max(piters) * 1.25)
# speedup annotation vs relTol=0.01
for i, t in enumerate(TOLS):
    sp = (1 - exec_t[i] / exec_t[0]) * 100
    if i: a.text(xs[i] - 0.18, exec_t[i] * 0.5, f"-{sp:.1f}%", ha="center", color="white", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{CASE}/plots/production-tol-forces.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130)
print("wrote", out)

# --- text summary: converged means + runtimes ---
print("\nrelTol |  Cd(mean) |  Cl(mean) |  Cs(mean) | ExecTime | p-iters/step | speedup")
for t in TOLS:
    d = data[t]; last = d[d[:, 0] >= d[-1, 0] - 300]
    cd, cl, cs = last[:, 1].mean(), last[:, 4].mean(), last[:, 10].mean()
    sp = (1 - RUNTIME[t][0] / RUNTIME["0.01"][0]) * 100
    print(f"  {t}  |  {cd:7.4f}  | {cl:8.4f}  | {cs:8.4f}  |  {RUNTIME[t][0]:6.0f}s |    {RUNTIME[t][2]:5.1f}     | -{sp:.1f}%")
