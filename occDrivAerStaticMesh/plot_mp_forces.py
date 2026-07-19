#!/usr/bin/env python3
"""Overlay Cd/Cl/Cs of the production mixed-precision variants (double / float /
mgsolver-float) from param-study-production-mp.sh. Left column = full history,
right column = zoomed to the common (shortest) window so the precision approaches
are directly comparable. Re-run to refresh.

Usage: python3 plot_mp_forces.py [run_stamp]   # default: newest -double/-float set
"""
import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

VARIANTS = [  # (label, run-dir suffix, colour)
    ("double (fp64 MG)", "double", "#1f77b4"),
    ("float (float32 MG precond)", "float", "#ff7f0e"),
]


def newest_dir(suffix):
    dirs = [d for d in glob.glob(os.path.join(ROOT, f"occDrivaerRun*-{suffix}"))
            if os.path.basename(d).endswith(f"-{suffix}")]
    return max(dirs, key=os.path.getmtime) if dirs else None


def load(suffix):
    """Newest occDrivaerRun*-<suffix> run dir with a non-empty coefficient.dat.
    Each variant is found independently, so variants rerun under different
    timestamps (e.g. float-only rerun) are still picked up."""
    dirs = [d for d in glob.glob(os.path.join(ROOT, f"occDrivaerRun*-{suffix}"))
            if os.path.basename(d).endswith(f"-{suffix}")]
    cands = []
    for d in dirs:
        f = glob.glob(os.path.join(d, "postProcessing", "neoForceCoeffs", "*",
                                   "coefficient.dat"))
        f = [x for x in f if os.path.getsize(x) > 0]
        if f:
            cands.append((os.path.getmtime(f[0]), os.path.basename(d), f[0]))
    if not cands:
        return None, None
    _, name, f = max(cands)
    data = np.loadtxt(f, comments="#", ndmin=2)
    return (data if data.size else None), name


def runtime(suffix, max_steps=900):
    """Parse the newest run log for (steps, ExecutionTime s, s/step, avg p-iters/step,
    failed), basing every statistic on only the first `max_steps` SIMPLE steps so the
    numbers match the plotted 0–max_steps force window (shorter runs use all steps)."""
    d = newest_dir(suffix)
    if not d:
        return None
    logs = glob.glob(os.path.join(d, "*.log"))
    if not logs:
        return None
    txt = open(max(logs, key=os.path.getmtime), errors="ignore").read().splitlines()
    step = 0
    et = None          # cumulative ExecutionTime at the end of step `max_steps`
    its = []           # pressure iters for solves within the first max_steps steps
    for ln in txt:
        if ln.startswith("Time = "):
            step += 1
            if step > max_steps:   # done: step max_steps fully seen (solves+exec)
                break
        if "ExecutionTime = " in ln:
            try:
                et = float(ln.split("ExecutionTime = ")[1].split(" s")[0])
            except (IndexError, ValueError):
                pass
        elif "Solving for p," in ln and "No Iterations" in ln:
            try:
                its.append(int(ln.split("No Iterations")[1].split()[0].rstrip(",")))
            except (IndexError, ValueError):
                pass
    steps = min(step, max_steps)   # actual steps the stats are based on
    piters = (sum(its) / len(its)) if its else None
    failed = any(k in "\n".join(txt) for k in ("NotSupported", "terminate", "signal "))
    perstep = (et / steps) if (et and steps) else None
    return dict(steps=steps, exec=et, perstep=perstep, piters=piters,
                failed=(failed and (steps < 5 or et is None)))


def main(argv):
    series = [("Cd (drag)", 1), ("Cl (lift)", 4), ("Cs (side)", 10)]
    loaded = []            # (label, col, data) -- variants with a force history
    rt = []                # (short label, suffix, colour, runtime dict) -- all variants
    for label, suf, col in VARIANTS:
        info = runtime(suf)
        rt.append((label.split(" (")[0], suf, col, info))
        d, name = load(suf)
        if d is None:
            tag = "FAILED" if (info and info["failed"]) else "no data"
            print(f"  [skip forces] {suf}: {tag}")
        else:
            loaded.append((label, col, d))
            print(f"  {label:34s} {len(d):5d} steps  ({name})  "
                  f"Cd={d[-1,1]:.4f} Cl={d[-1,4]:.4f} Cs={d[-1,10]:.4f}")
        if info:
            ps = f"{info['perstep']:.3f}s/step" if info["perstep"] else "n/a"
            pit = f"{info['piters']:.1f}" if info["piters"] else "n/a"
            print(f"        runtime: {info['steps']} steps, exec={info['exec']}s, "
                  f"{ps}, p-iters/step={pit}, failed={info['failed']}")
    if not loaded:
        print("no variant has force data"); return 1

    XMAX = 900   # show only the full history up to this SIMPLE iteration

    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.32, wspace=0.22)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]

    # --- force history (Cd/Cl/Cs), full history clipped to XMAX, side by side ---
    for col_i, (lab, ci) in enumerate(series):
        ax = axes[col_i]
        for label, col, d in loaded:
            m = d[:, 0] <= XMAX
            ax.plot(d[m, 0], d[m, ci], color=col, lw=1.0, label=label)
        ax.set_ylabel(lab)
        ax.set_xlim(0, XMAX)
        ax.set_xlabel("SIMPLE iteration")
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"Full history (0–{XMAX} steps)", fontsize=11)
    axes[0].legend(fontsize=8.5, loc="best")

    # --- performance panel: wall time per step + p-iters/step, over first XMAX steps ---
    axp = fig.add_subplot(gs[1, :])
    axt = axp.twinx()
    x = np.arange(len(rt)); bw = 0.36
    persteps = [(i["perstep"] if i and i["perstep"] and not i["failed"] else 0) for _, _, _, i in rt]
    piters = [(i["piters"] if i and i["piters"] and not i["failed"] else 0) for _, _, _, i in rt]
    cols = [c for _, _, c, _ in rt]
    mx = max(persteps) if any(persteps) else 1.0
    base = next((p for p in persteps if p), None)
    axp.bar(x - bw/2, persteps, bw, color=cols, label="wall time / step")
    axt.bar(x + bw/2, piters, bw, color="0.6", alpha=0.7, label="p-iters / step")
    axp.set_xticks(x); axp.set_xticklabels([lab for lab, _, _, _ in rt], fontsize=9)
    axp.set_ylabel("wall time / SIMPLE step [s]"); axt.set_ylabel("pressure iters / step")
    axp.set_ylim(0, mx * 1.4); axt.set_ylim(0, (max(piters) if any(piters) else 1) * 1.4)
    axp.grid(True, axis="y", alpha=0.3)
    axp.set_title("Performance over first 900 steps — wall time per step  (left bars)  "
                  "+  pressure iters per step  (grey, right)", fontsize=11)
    for i, (lab, _, _, info) in enumerate(rt):
        if persteps[i]:
            t = f"{persteps[i]:.3f} s/st"
            if base and persteps[i] != base:
                t += f"\n{(persteps[i]/base - 1)*100:+.1f}%"
            axp.text(x[i]-bw/2, persteps[i] + mx*0.02, t, ha="center", va="bottom",
                     fontsize=8.5, fontweight="bold")
            axp.text(x[i]-bw/2, mx*0.05, f"{info['steps']} st\n{info['exec']:.0f}s tot",
                     ha="center", va="bottom", fontsize=7.5, color="white")
        else:
            axp.text(x[i], mx*0.5, "FAILED" if (info and info["failed"]) else "no run",
                     ha="center", va="center", fontsize=12, color="crimson", fontweight="bold")
        if piters[i]:
            axt.text(x[i]+bw/2, piters[i] + axt.get_ylim()[1]*0.01, f"{piters[i]:.1f}",
                     ha="center", va="bottom", fontsize=8.5, color="0.3")

    desc = "   ".join(f"{lab}: {info['exec']:.0f}s / {info['steps']} st"
                      for lab, _, _, info in rt if info and info["exec"])
    fig.suptitle("Production forces + performance — precision comparison "
                 "(first 900 steps)\n" + desc, fontweight="bold", fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.86, bottom=0.09)
    p = os.path.join(OUT, "7_mp_force_comparison.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote:", os.path.relpath(p, HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
