#!/usr/bin/env python3
"""Peek at a paper-study results grid, reporting STEADY-STATE s/step.

WHY THIS EXISTS: the obvious metric, ExecutionTime_final / steps, silently includes the ~47 s of
startup (mesh load, decomposition, solver setup) amortized over the window -- inflating a 50-step
run by ~0.9 s/step (e.g. a true 2.377 became a reported 3.300). Startup turned out to be near
constant across configs (47.4-48.7 s over L3..L6, both sc branches), so RANKINGS from the naive
metric held, but every absolute number was ~40 % too high and the relative gaps were compressed.
Steady state = (ET_last - ET_first) / (steps - 1)  -- excludes setup and the first step.

Usage:
  ./peek-grid.py                          # default: the no-sc rel-tol grid
  ./peek-grid.py mgsc-coarse-reltol       # the sc=ON grid
  ./peek-grid.py mgnosc-coarse-reltol --levels 2 3 4 5 6 8
  ./peek-grid.py <dir> --fields           # add a per-field solve breakdown for the best cell
"""
import argparse
import glob
import os
import re

RESULTS = "paperParamStudyResults"
# tol -> variant-name code (must match phase3h's tol_code)
TOL_CODE = {"0.25": "025", "0.2": "02", "0.15": "015",
            "0.1": "1", "0.01": "2", "0.001": "3", "0.0001": "4"}


def metrics(path):
    """(steps, steady_s_per_step, naive, startup, mean_outer_iters, p_ms, cont) or None."""
    try:
        lines = open(path, errors="ignore").readlines()
    except OSError:
        return None
    steps = sum(1 for l in lines if l.startswith("Time = "))
    et = [float(m.group(1)) for l in lines
          if (m := re.search(r"ExecutionTime = ([0-9.]+) s", l))]
    if steps < 2 or len(et) < 2:
        return None
    steady = (et[-1] - et[0]) / (steps - 1)
    naive = et[-1] / steps
    its, pms, n = 0, 0.0, 0
    for l in lines:
        if "Solving for p," in l:
            if (m := re.search(r"No Iterations (\d+)", l)):
                its += int(m.group(1))
                n += 1
            if (m := re.search(r"Solve time = ([0-9.]+)", l)):
                pms += float(m.group(1))
    cont = None
    for l in reversed(lines):
        if (m := re.search(r"sum local = ([0-9.eE+-]+)", l)):
            cont = float(m.group(1))
            break
    return (steps, steady, naive, et[0],
            its / n if n else None, pms / n if n else None, cont)


def field_breakdown(path, steady):
    """Per-field mean solve ms/step and % of the step."""
    fields = {}
    lines = open(path, errors="ignore").readlines()
    steps = sum(1 for l in lines if l.startswith("Time = "))
    for l in lines:
        if (m := re.search(r"Solving for (\w+),.*Solve time = ([0-9.]+)", l)):
            fields.setdefault(m.group(1), []).append(float(m.group(2)))
    out = []
    for fld, ts in fields.items():
        per_step = sum(ts) / steps           # ms per step
        out.append((fld, len(ts), sum(ts) / len(ts), per_step, per_step / 1000 / steady * 100))
    return out, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", nargs="?", default="mgnosc-coarse-reltol")
    ap.add_argument("--levels", nargs="*", default=["2", "3", "4", "5", "6", "8"])
    ap.add_argument("--tols", nargs="*",
                    default=["0.25", "0.2", "0.15", "0.1", "0.01", "0.001", "0.0001"])
    ap.add_argument("--fields", action="store_true",
                    help="also print a per-field solve breakdown for the best cell")
    ap.add_argument("--metric", choices=["pms", "steady", "iters"], default="pms",
                    help="cell metric. pms (default) = mean pressure-solve ms -- the quantity these "
                         "sweeps actually change. steady = s/step, which DILUTES the signal ~10x "
                         "because momentum/turbulence/assemble are identical across cells (a 15%% "
                         "p_ms change reads as ~1.5%% on the step). iters = outer CG count.")
    a = ap.parse_args()
    d = os.path.join(RESULTS, a.study)
    if not os.path.isdir(d):
        print(f"!! no such study dir: {d}")
        return 1

    MIDX = {"steady": 1, "iters": 4, "pms": 5}[a.metric]
    MFMT = {"steady": "{:.3f}", "iters": "{:.1f}", "pms": "{:.0f}"}[a.metric]
    MLAB = {"steady": "steady s/step", "iters": "outer iters",
            "pms": "mean p-solve [ms]"}[a.metric]
    best = (1e9, None, None)
    grid = {}
    for L in a.levels:
        for t in a.tols:
            code = TOL_CODE.get(t)
            g = sorted(glob.glob(f"{d}/L{L}e{code}-2026*.log"), reverse=True)
            if not g:
                continue
            m = metrics(g[0])
            if not m:
                continue
            grid[(L, t)] = (m, g[0])
            if m[0] >= 50 and m[MIDX] is not None and m[MIDX] < best[0]:  # full windows only
                best = (m[MIDX], (L, t), g[0])

    print(f"=== {a.study}: {MLAB}  (cell = {a.metric} / outer-iters, ~n = partial) ===")
    hdr = "L\\tol"
    print(f"{hdr:6s}" + "".join(f"{t:>13s}" for t in a.tols))
    for L in a.levels:
        row = f"{'L'+L:6s}"
        for t in a.tols:
            if (L, t) not in grid:
                row += f"{'.':>13s}"
                continue
            cellm, _ = grid[(L, t)]
            val = cellm[MIDX]
            its = cellm[4]
            tag = f"~{cellm[0]}" if cellm[0] < 50 else ""
            star = "*" if best[1] == (L, t) else ""
            row += f"{MFMT.format(val):>6s}/{its:<4.1f}{tag}{star:1s}"[:13].rjust(13)
        print(row)

    if best[1]:
        (steps, steady, naive, startup, its, pms, cont), path = grid[best[1]]
        print(f"\nbest by {a.metric}: L{best[1][0]} tol={best[1][1]}")
        print(f"      p_solve={pms:.0f} ms   outer-iters={its:.1f}   steady={steady:.3f} s/step   "
              f"cont={cont:.2e}")
        print(f"      (naive s/step with startup would read {naive:.3f}; startup={startup:.1f}s)")
        if a.fields:
            fb, nst = field_breakdown(path, steady)
            print(f"\n  per-field solve breakdown ({nst} steps, step={steady:.3f}s):")
            print(f"  {'field':6s} {'solves':>7s} {'mean ms':>9s} {'ms/step':>9s} {'% step':>8s}")
            tot = 0.0
            for fld, n, mean, per_step, pct in fb:
                tot += pct
                print(f"  {fld:6s} {n:7d} {mean:9.1f} {per_step:9.1f} {pct:7.1f}%")
            print(f"  {'ALL':6s} {'':7s} {'':9s} {'':9s} {tot:7.1f}%   "
                  f"(non-solve remainder: {100-tot:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
