#!/usr/bin/env python3
"""Generate performance plots for the occDrivAerStaticMesh solver study.

Reads the per-run logs under paramStudyResults/<study>/ (written by the param-study
harness) and produces a focused set of PNGs under plots/. Each plot is built from a
single internally-consistent study so the comparison is apples-to-apples. Re-run after
new sweeps to refresh.

Metrics parsed per log (matching param-study-table.sh):
  exec      last 'ExecutionTime = X s'           (total solver time, s)
  p_iters   mean 'No Iterations N' on 'Solving for p,' lines
  p_ms      mean 'Solve time = X ms' on those lines
  steps     count of 'Time = N' lines

Usage:  python3 plot_performance.py
"""
import re
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "paramStudyResults")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

STAMP_RE = re.compile(r"-\d{8}-\d{6}\.log$")
PITER_RE = re.compile(r"Solving for p,.*No Iterations\s+(\d+)")
PMS_RE = re.compile(r"Solving for p,.*Solve time =\s+([\d.]+)\s*ms")
EXEC_RE = re.compile(r"ExecutionTime = ([\d.]+)\s*s")
STEP_RE = re.compile(r"^Time = ", re.M)

# Highlight colour for the winning / best-practice config.
HL = "#d62728"
BASE = "#1f77b4"
ALT = "#ff7f0e"


def newest_log(study, name):
    """Newest timestamped log for a stripped run-name in a study dir."""
    cands = sorted(
        glob.glob(os.path.join(RESULTS, study, f"{name}-*.log")),
        key=os.path.getmtime,
        reverse=True,
    )
    return cands[0] if cands else None


def parse(log):
    """Return dict(exec, p_iters, p_ms, steps) or None if unreadable/empty."""
    if not log or not os.path.isfile(log):
        return None
    txt = open(log, errors="ignore").read()
    steps = len(STEP_RE.findall(txt))
    execs = EXEC_RE.findall(txt)
    its = [int(x) for x in PITER_RE.findall(txt)]
    ms = [float(x) for x in PMS_RE.findall(txt)]
    if not execs or steps == 0:
        return None
    return {
        "exec": float(execs[-1]),
        "p_iters": sum(its) / len(its) if its else float("nan"),
        "p_ms": sum(ms) / len(ms) if ms else float("nan"),
        "steps": steps,
    }


def collect(study, names, min_steps=30):
    """names: list of (label, run-name). Returns list of (label, metrics)."""
    out = []
    for label, run in names:
        m = parse(newest_log(study, run))
        if m and m["steps"] >= min_steps:
            out.append((label, m))
        else:
            print(f"  [skip] {study}/{run}: "
                  + ("partial/empty" if not m else f"only {m['steps']} steps"))
    return out


# --------------------------------------------------------------------------- #
# Plot 1 — coarse-grid-solver sweep (the precfloat float-MG investigation)
# --------------------------------------------------------------------------- #
def plot_coarse_solve():
    iters = [1, 5, 10, 15, 20, 25, 50]
    series = {
        "CG + Jacobi": ("cg", BASE, "o"),
        "damped Jacobi (Ir)": ("jacobi", ALT, "s"),
    }
    data = {}
    for lbl, (tag, _c, _m) in series.items():
        rows = []
        for n in iters:
            m = parse(newest_log(
                "coarse-solve",
                f"pMG-precfloat-coarse-{tag}{n}-cache-rebuild100"))
            rows.append(m)
        data[lbl] = rows

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for lbl, (tag, c, mk) in series.items():
        ex = [r["exec"] if r else None for r in data[lbl]]
        it = [r["p_iters"] if r else None for r in data[lbl]]
        ax1.plot(iters, ex, marker=mk, color=c, label=lbl)
        ax2.plot(iters, it, marker=mk, color=c, label=lbl)

    # mark the winner (CG, 10 iters)
    cg = data["CG + Jacobi"]
    if cg[2]:
        ax1.scatter([10], [cg[2]["exec"]], s=160, facecolors="none",
                    edgecolors=HL, linewidths=2, zorder=5)
        ax1.annotate("cg10\n(best)", (10, cg[2]["exec"]),
                     textcoords="offset points", xytext=(6, 14), color=HL,
                     fontsize=9, fontweight="bold")

    ax1.set_xlabel("coarse-solver iterations")
    ax1.set_ylabel("total ExecutionTime (s)  —  30 steps")
    ax1.set_title("End-to-end runtime")
    ax2.set_xlabel("coarse-solver iterations")
    ax2.set_ylabel("mean outer pressure iterations / solve")
    ax2.set_title("Outer pressure-solver work")
    for ax in (ax1, ax2):
        ax.set_xticks(iters)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Coarse-grid solver sweep — localized float-MG preconditioner "
                 "(CG beats Jacobi; saturates at 10 iters)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, "1_coarse_solve_sweep.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Plot 2 — MG depth sweep (localized): L10 is the sweet spot
# --------------------------------------------------------------------------- #
def plot_mg_depth():
    levels = [2, 4, 6, 8, 10, 15, 20]
    rows = collect("mg-tuning",
                   [(L, f"pMG-localized-L{L}-ukoSmooth") for L in levels])
    Ls = [int(l) for l, _ in rows]
    ex = [m["exec"] for _, m in rows]
    it = [m["p_iters"] for _, m in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.6))
    ax1.plot(Ls, ex, marker="o", color=BASE, label="ExecutionTime")
    ax1.set_xlabel("MG max_levels")
    ax1.set_ylabel("total ExecutionTime (s)", color=BASE)
    ax1.tick_params(axis="y", labelcolor=BASE)
    ax2 = ax1.twinx()
    ax2.plot(Ls, it, marker="s", color=ALT, label="p-iters/solve")
    ax2.set_ylabel("mean pressure iterations / solve", color=ALT)
    ax2.tick_params(axis="y", labelcolor=ALT)

    if 10 in Ls:
        i = Ls.index(10)
        ax1.scatter([10], [ex[i]], s=170, facecolors="none", edgecolors=HL,
                    linewidths=2, zorder=5)
        ax1.annotate("L10 sweet spot", (10, ex[i]), textcoords="offset points",
                     xytext=(8, 16), color=HL, fontsize=10, fontweight="bold")
    ax1.set_xticks(Ls)
    ax1.grid(True, alpha=0.3)
    fig.suptitle("MG depth sweep (localized) — deeper helps until L10, then flat",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, "2_mg_depth_sweep.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Plot 3 — variant comparison at L10: why "localized"
# --------------------------------------------------------------------------- #
def plot_variants():
    names = [
        ("localized\n(Schwarz block-Jacobi)", "pMG-localized-L10-ukoSmooth"),
        ("global / non-localized", "pMG-L10-sc0-ukoSmooth"),
        ("FCG + MG precond", "pFCG-MGprec-ukoSmooth"),
        ("CG-coarse solver", "pMG-cgcoarse-ukoSmooth"),
        ("2 smoothing steps", "pMG-smooth2-ukoSmooth"),
    ]
    rows = collect("mg-tuning", names)
    labels = [l for l, _ in rows]
    ex = [m["exec"] for _, m in rows]
    it = [m["p_iters"] for _, m in rows]
    order = sorted(range(len(ex)), key=lambda i: ex[i])
    labels = [labels[i] for i in order]
    ex = [ex[i] for i in order]
    it = [it[i] for i in order]
    colors = [HL if "localized\n" in l else BASE for l in labels]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(range(len(ex)), ex, color=colors)
    for i, (b, n) in enumerate(zip(bars, it)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{ex[i]:.0f}s\n{n:.0f} p-it", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("total ExecutionTime (s)")
    ax.set_ylim(0, max(ex) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Why localized — fewest-iterations isn't fastest (at L10)",
                 fontweight="bold")
    ax.set_title("localized trades more outer iters for far cheaper per-iter cost",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, "3_localized_vs_variants.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Plot 4 — preconditioner headline (mg-headline, same session)
# --------------------------------------------------------------------------- #
def plot_preconditioner():
    names = [
        ("plain global MG", "pMG-ukoSmooth"),
        ("localized MG", "pMG-localized-ukoSmooth"),
        ("scalecorr (global)", "pMG-scalecorr-ukoSmooth"),
        ("float-MG + scalecorr\n+ cache", "pMG-precfloat-sc-cache-rebuild100-ukoSmooth"),
        ("scalecorr + localized\n(anti-pattern)", "pMG-scale-correction-localized-ukoSmooth"),
    ]
    rows = collect("mg-headline", names)
    labels = [l for l, _ in rows]
    ex = [m["exec"] for _, m in rows]
    it = [m["p_iters"] for _, m in rows]
    colors = []
    for l in labels:
        if "anti-pattern" in l:
            colors.append("#7f7f7f")
        elif "float-MG" in l or l == "localized MG":
            colors.append(HL)
        else:
            colors.append(BASE)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(range(len(ex)), ex, color=colors)
    ax.set_yscale("log")
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.03,
                f"{ex[i]:.0f}s\n{it[i]:.0f} p-it", ha="center", va="bottom",
                fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("total ExecutionTime (s, log scale)")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.suptitle("Preconditioner comparison — localized wins, scalecorr+localized "
                 "is a 9× trap", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, "4_preconditioner_headline.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Plot 5 — force-coefficient convergence history (neoForceCoeffs output)
# --------------------------------------------------------------------------- #
def find_coeff_dat():
    """Newest postProcessing/neoForceCoeffs/<t0>/coefficient.dat under any
    occDrivaerRun*/ beside the case dir (RUN_ROOT = parent of the case)."""
    root = os.path.dirname(HERE)
    cands = glob.glob(os.path.join(
        root, "occDrivaerRun*", "postProcessing", "neoForceCoeffs",
        "*", "coefficient.dat"))
    cands = [c for c in cands if os.path.getsize(c) > 0]
    return max(cands, key=os.path.getmtime) if cands else None


def plot_force_coeffs(datfile=None, avg_window=1800, annotate_mean=True,
                      outname="5_force_coefficients.png", history_only=False):
    datfile = datfile or find_coeff_dat()
    if not datfile or not os.path.isfile(datfile):
        print("  [skip] no neoForceCoeffs coefficient.dat found")
        return None
    # columns: 0 Time 1 Cd 2 Cd(f) 3 Cd(r) 4 Cl 5 Cl(f) 6 Cl(r)
    #          7 CmPitch 8 CmRoll 9 CmYaw 10 Cs 11 Cs(f) 12 Cs(r)
    data = np.loadtxt(datfile, comments="#", ndmin=2)
    t = data[:, 0]
    coeffs = [("Cd (drag)", data[:, 1], BASE),
              ("Cl (lift)", data[:, 4], ALT),
              ("Cs (side)", data[:, 10], "#2ca02c")]
    # averaging window: the LAST `avg_window` timesteps (clamped to the run length
    # for shorter runs).
    avg_start = max(t[0], t[-1] - avg_window + 1)
    win = t >= avg_start
    navg = int(win.sum())

    ncols = 1 if history_only else 2
    fig, axes = plt.subplots(3, ncols, figsize=(6.5 if history_only else 13, 7.5),
                             sharex="col", squeeze=False)
    for row, (label, y, c) in enumerate(coeffs):
        # left column: full convergence history (no averages shown)
        axL = axes[row, 0]
        axL.plot(t, y, color=c, lw=1.0)
        axL.set_ylabel(label, color=c)
        axL.tick_params(axis="y", labelcolor=c)
        axL.grid(True, alpha=0.3)

        if history_only:
            continue
        # right column: values + average over iterations >= avg_start
        axR = axes[row, 1]
        axR.plot(t[win], y[win], color=c, lw=1.0)
        mean = y[win].mean()
        axR.axhline(mean, color=c, ls="--", lw=1, alpha=0.7)
        if annotate_mean:
            axR.annotate(f"mean(last {navg}) = {mean:.4f}",
                         xy=(t[win][-1], mean), xytext=(-8, 6),
                         textcoords="offset points", ha="right", color=c,
                         fontsize=9, fontweight="bold")
        axR.tick_params(axis="y", labelcolor=c)
        axR.grid(True, alpha=0.3)

    axes[0, 0].set_title("Full convergence history", fontsize=10)
    axes[-1, 0].set_xlabel("SIMPLE iteration")
    if not history_only:
        axes[0, 1].set_title(f"Last {navg} timesteps (with average)", fontsize=10)
        axes[-1, 1].set_xlabel("SIMPLE iteration")
    # datfile = <run>/postProcessing/neoForceCoeffs/<t0>/coefficient.dat -> up 4 dirs to <run>
    run = os.path.basename(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(datfile)))))
    fig.suptitle(f"Force-coefficient convergence — neoForceCoeffs ({run})\n"
                 f"{len(t)} iterations, occDrivAerStaticMesh production run",
                 fontweight="bold", fontsize=10 if history_only else 12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, outname)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    navg = int(win.sum())
    print(f"  source: {os.path.relpath(datfile, HERE)}  "
          f"(averaging window: {navg} iters from {avg_start})")
    return p


def _autocorr_tau(x):
    """Integrated autocorrelation time tau_int = 0.5 + sum_{k>=1} rho_k, using the
    initial-positive-sequence estimator (truncate the sum at the first non-positive
    rho_k). For an uncorrelated series tau_int -> 0.5, so N_eff = N/(2*tau) -> N."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0.0:
        return 0.5
    tau = 0.5
    for k in range(1, n):
        rho = np.dot(x[:-k], x[k:]) / (n * var)
        if rho <= 0.0:            # truncate at first non-positive autocorrelation
            break
        tau += rho
    return tau


def plot_force_coeffs_ci(datfile=None, avg_window=1800, conf=0.95,
                         outname="5_force_coefficients_ci.png"):
    """Windowed force coefficients with the mean and an autocorrelation-corrected
    confidence interval on the mean. The CI uses the effective sample size
    N_eff = N/(2*tau_int) (tau_int = integrated autocorrelation time), which is the
    statistically honest interval for a correlated CFD time series — a naive
    sigma/sqrt(N) band would be ~sqrt(2*tau) times too narrow."""
    datfile = datfile or find_coeff_dat()
    if not datfile or not os.path.isfile(datfile):
        print("  [skip] no neoForceCoeffs coefficient.dat found")
        return None
    data = np.loadtxt(datfile, comments="#", ndmin=2)
    t = data[:, 0]
    coeffs = [("Cd (drag)", data[:, 1], BASE),
              ("Cl (lift)", data[:, 4], ALT),
              ("Cs (side)", data[:, 10], "#2ca02c")]
    avg_start = max(t[0], t[-1] - avg_window + 1)
    win = t >= avg_start
    navg = int(win.sum())
    tw = t[win]
    z = 1.959963985 if abs(conf - 0.95) < 1e-9 else \
        float(np.sqrt(2) * _erfinv_safe(conf))

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True, squeeze=False)
    for row, (label, y, c) in enumerate(coeffs):
        yw = y[win]
        mean = yw.mean()
        sd = yw.std(ddof=1)
        tau = _autocorr_tau(yw)
        n_eff = navg / (2.0 * tau)
        sem = sd / np.sqrt(n_eff)          # autocorrelation-corrected std error
        ci = z * sem
        ax = axes[row, 0]
        ax.plot(tw, yw, color=c, lw=0.9, alpha=0.8)
        ax.axhline(mean, color=c, ls="-", lw=1.4)
        ax.axhspan(mean - ci, mean + ci, color=c, alpha=0.18, lw=0)
        ax.set_ylabel(label, color=c)
        ax.tick_params(axis="y", labelcolor=c)
        ax.grid(True, alpha=0.3)
        ax.annotate(
            f"mean = {mean:.4f}  ±{ci:.4f}  ({int(conf*100)}% CI)\n"
            f"$\\tau_{{int}}$ = {tau:.1f},  N_eff = {n_eff:.0f} / {navg}",
            xy=(0.015, 0.04), xycoords="axes fraction", ha="left", va="bottom",
            color=c, fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c, alpha=0.85))
    axes[-1, 0].set_xlabel("SIMPLE iteration")
    run = os.path.basename(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(datfile)))))
    fig.suptitle(
        f"Force coefficients — mean ± {int(conf*100)}% CI over last {navg} "
        f"iterations ({run})\n"
        f"CI corrected for autocorrelation via effective sample size "
        f"N_eff = N/(2$\\tau_{{int}}$)",
        fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(OUT, outname)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  source: {os.path.relpath(datfile, HERE)}  "
          f"(CI window: {navg} iters from {avg_start})")
    return p


def _erfinv_safe(conf):
    """z for a two-sided CI without scipy: invert the normal CDF via a rational
    approximation of the inverse error function (Winitzki), good to ~1e-3."""
    p = conf + (1.0 - conf) / 2.0          # one-sided tail prob
    x = 2.0 * p - 1.0
    a = 0.147
    ln = np.log(1 - x * x)
    tt = 2 / (np.pi * a) + ln / 2
    return np.sign(x) * np.sqrt(np.sqrt(tt * tt - ln / a) - tt)


# --------------------------------------------------------------------------- #
# Plot 6 — computational-cost breakdown (kokkos space-time-stack profile)
# --------------------------------------------------------------------------- #
TREE_RE = re.compile(
    r"^(?P<pre>[|>\s-]*?)->\s+(?P<t>[\d.eE+-]+)\s+sec\s+(?P<pct>[\d.]+)%"
    r".*?\s(?P<name>\S+)\s+\[(?:region|for)\]")


def find_profile():
    """Newest best-practice localized-MG kokkos space-time-stack profile."""
    cands = glob.glob(os.path.join(
        RESULTS, "best-practice",
        "pMG-localized-best*kokkos-profile.txt"))
    return max(cands, key=os.path.getmtime) if cands else None


def parse_topdown(prof):
    """Parse the TOP-DOWN TIME TREE into a flat list of (depth, time, pct, name).
    depth = number of '|' before the '->' arrow."""
    rows, total, in_tree = [], None, False
    for ln in open(prof, errors="ignore"):
        if ln.startswith("TOTAL TIME:"):
            total = float(ln.split(":")[1].split()[0])
        if ln.startswith("TOP-DOWN TIME TREE"):
            in_tree = True; continue
        if ln.startswith("BOTTOM-UP TIME TREE"):
            break
        if not in_tree:
            continue
        m = TREE_RE.match(ln)
        if m:
            depth = ln[:ln.index("->")].count("|")
            rows.append((depth, float(m["t"]), float(m["pct"]),
                         m["name"].split(".")[-1]))
    return rows, total


def children_of(rows, idx):
    """Direct children (depth+1) of rows[idx], until depth drops back."""
    d0 = rows[idx][0]
    out = []
    for d, t, p, n in rows[idx + 1:]:
        if d <= d0:
            break
        if d == d0 + 1:
            out.append((d, t, p, n))
    return out


def plot_cost_breakdown():
    prof = find_profile()
    if not prof:
        print("  [skip] no kokkos-profile.txt found"); return None
    rows, total = parse_topdown(prof)
    by = lambda name, depth=None: next(
        (r for r in rows if r[3] == name and (depth is None or r[0] == depth)), None)

    # --- pressure-corrector internals (used for both the donut split and panel B) ---
    pe_idx = next((i for i, r in enumerate(rows) if r[3] == "pEqn"), None)
    pe = rows[pe_idx]
    named = {n: t for _, t, _, n in children_of(rows, pe_idx)}
    solversetup = named.get("solverSetup", 0)
    createmtx = named.get("createMtx", 0)
    assemble = named.get("spatialImplicit", 0)
    mg_solve = pe[1] - solversetup - createmtx - assemble
    pcorr = by("pressureCorrector", 2) or by("pressureCorrector", 1)
    pcorr_other = (pcorr[1] - pe[1]) if pcorr else 0      # HbyA/flux/grad/relax

    # --- momentum-predictor internals: direct children = assembly; remainder = solve ---
    mp_idx = next(i for i, r in enumerate(rows)
                  if r[3] == "momentumPredictor" and r[0] == 2)
    mp = rows[mp_idx]
    mom_rest = sum(t for _, t, _, _ in children_of(rows, mp_idx))   # assemble/construct/createMtx
    mom_solve = mp[1] - mom_rest                                    # the U linear solve

    # --- donut phases: pressure & momentum each split solve / assembly; write EXCLUDED ---
    pres_solve = mg_solve + solversetup                            # the Ginkgo MG solve
    pres_rest = pcorr[1] - pres_solve                             # assembly + flux/grad/HbyA
    turb = by("turbulenceCorrect", 2)
    phases = [
        ("pressure solve", pres_solve, HL),
        ("pressure assembly/rest", pres_rest, "#f2a6a6"),
        ("momentum solve", mom_solve, "#1f77b4"),
        ("momentum assembly/rest", mom_rest, "#aec7e8"),
        ("turbulence (k+ω)", turb[1] if turb else 0, "#2ca02c"),
    ]
    phases = [(n, t, c) for n, t, c in phases if t > 0.05]
    step_total = sum(t for _, t, _ in phases)
    labels = [n for n, _, _ in phases]
    sizes = [t for _, t, _ in phases]
    pcolors = [c for _, _, c in phases]

    # --- pressure-solve internals for panel B ---
    sub = [("MG V-cycle solve", mg_solve, "#1f77b4"),
           ("solverSetup\n(hierarchy build)", solversetup, "#ff7f0e"),
           ("createMtx (CSR)", createmtx, "#2ca02c"),
           ("assemble", assemble, "#9467bd"),
           ("HbyA/flux/grad/relax", pcorr_other, "#8c8c8c")]
    sub = [(n, t, c) for n, t, c in sub if t > 0.05]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.6),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    # Panel A: phase donut. Legend sits BELOW the donut (not in the hole, which
    # overlapped the ring); the hole shows the total only.
    wedges, _ = axA.pie(sizes, colors=pcolors, startangle=90,
                        wedgeprops=dict(width=0.42, edgecolor="w"))
    axA.text(0, 0, f"{step_total:.0f}s\nper-step\ntotal", ha="center", va="center",
             fontsize=14, fontweight="bold")
    axA.legend(wedges, [f"{l.replace(chr(10), ' ')} — {s:.0f}s ({100*s/step_total:.0f}%)"
                        for l, s in zip(labels, sizes)],
               loc="upper center", bbox_to_anchor=(0.5, -0.02),
               fontsize=11, frameon=False, ncol=2)
    axA.set_title(f"Per-step phase breakdown — {step_total:.0f}s / 30 steps",
                  fontsize=13)

    # Panel B: pressure-solve internals (horizontal stacked bar)
    left = 0
    for n, t, c in sub:
        axB.barh(0, t, left=left, color=c, edgecolor="w", label=f"{n} — {t:.1f}s")
        if t > step_total * 0.02:
            axB.text(left + t / 2, 0, f"{t:.0f}s", ha="center", va="center",
                     color="w", fontsize=11, fontweight="bold")
        left += t
    axB.set_xlim(0, pcorr[1] if pcorr else left)
    axB.set_ylim(-1, 1.4)
    axB.set_yticks([])
    axB.set_xlabel("seconds (of the pressure-corrector phase)", fontsize=12)
    axB.tick_params(labelsize=11)
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2,
               fontsize=10, frameon=False)
    axB.set_title(f"Inside the pressure corrector ({pcorr[1]:.0f}s)", fontsize=13)

    fig.suptitle("Computational-cost breakdown — best-practice localized MG "
                 "(Kokkos space-time-stack)", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, "6_cost_breakdown.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  source: {os.path.relpath(prof, HERE)}")
    return p


if __name__ == "__main__":
    made = []
    # optional CLI arg: explicit coefficient.dat path for plot 5
    coeff_arg = sys.argv[1] if len(sys.argv) > 1 else None
    jobs = [
        ("plot_coarse_solve", plot_coarse_solve),
        ("plot_mg_depth", plot_mg_depth),
        ("plot_variants", plot_variants),
        ("plot_preconditioner", plot_preconditioner),
        ("plot_force_coeffs", lambda: plot_force_coeffs(coeff_arg)),
        ("plot_cost_breakdown", plot_cost_breakdown),
    ]
    for name, fn in jobs:
        print(f"== {name} ==")
        try:
            made.append(fn())
        except Exception as e:  # keep going if one study's data is missing
            print(f"  !! {name} failed: {e}")
    print("\nwrote:")
    for p in made:
        print("  " + os.path.relpath(p, HERE))
