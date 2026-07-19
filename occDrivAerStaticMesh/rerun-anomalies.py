#!/usr/bin/env python3
"""Detect transient-inflated cells in a rel-tol grid, delete their logs, and print the re-run list.

WHY: a handful of cells per grid come back with ~2x their row's iteration count while every neighbour
is healthy (seen repeatedly this session; every one re-run so far dropped into line). The signature is
discontinuity with the WHOLE row, so the row median is the right reference -- not the global best, and
not a fixed threshold, since iteration counts differ by an order of magnitude across depths.

Detection: compare each cell to its IMMEDIATE NEIGHBOURS in the row (adjacent tolerances), not to the
row median. Two reasons the median fails: (a) iteration count legitimately TRENDS across tolerance, so
the median is not a flat baseline; (b) the anomalies inflate the very median used to detect them -- with
2 of 6 cells bad, L2's median rose to 17.3 and a genuine 24.1 outlier scored only 1.39x and escaped.
A transient is discontinuous with BOTH its neighbours, so neighbours are the right reference.
Default factor 1.5: far above run-to-run noise (~1%) and below every observed anomaly (1.8-2.6x).

Usage:
  ./rerun-anomalies.py <study> [--factor 1.5] [--delete]
     without --delete: report only (dry run)
     with --delete:    remove the offending logs so run_one will re-run them
Prints the LEVELS/TOLS needed to re-run, for feeding back into phase3h/phase3i.
"""
import argparse
import glob
import os
import re
import statistics
import sys

RESULTS = "paperParamStudyResults"
CODE2TOL = {"025": "0.25", "02": "0.2", "015": "0.15", "1": "0.1", "2": "0.01", "3": "0.001",
            "4": "0.0001"}


def cell_iters(path):
    """(mean outer iters, steps) or None."""
    try:
        lines = open(path, errors="ignore").readlines()
    except OSError:
        return None
    steps = sum(1 for l in lines if l.startswith("Time = "))
    its = [int(m.group(1)) for l in lines if "Solving for p," in l
           and (m := re.search(r"No Iterations (\d+)", l))]
    if not its:
        return None
    return statistics.mean(its), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    d = os.path.join(RESULTS, a.study)
    if not os.path.isdir(d):
        print(f"!! no such study: {d}")
        return 1

    # gather: rows[level][tol] = (iters, steps, path)
    rows = {}
    for f in glob.glob(f"{d}/L*e*-2026*.log"):
        m = re.match(r"L(\d+)e(\d+)$", os.path.basename(f).split("-2026")[0])
        if not m:
            continue
        lvl, code = int(m.group(1)), m.group(2)
        tol = CODE2TOL.get(code)
        if tol is None:
            continue
        c = cell_iters(f)
        if not c or c[1] < 50:          # ignore partials -- they are not evidence of anything
            continue
        # keep the newest log per cell
        prev = rows.setdefault(lvl, {}).get(tol)
        if prev is None or f > prev[2]:
            rows[lvl][tol] = (c[0], c[1], f)

    bad = []
    for lvl in sorted(rows):
        # order the row by tolerance (loose -> tight), the axis along which iters trend smoothly
        ordered = sorted(rows[lvl].items(), key=lambda kv: -float(kv[0]))
        if len(ordered) < 3:
            continue
        for i, (tol, (its, steps, path)) in enumerate(ordered):
            nbrs = []
            if i > 0:
                nbrs.append(ordered[i - 1][1][0])
            if i < len(ordered) - 1:
                nbrs.append(ordered[i + 1][1][0])
            # a neighbour that is itself anomalous would mask this one; use the MIN neighbour, which
            # is the healthy side of any adjacent pair
            ref = min(nbrs)
            if its > a.factor * ref:
                bad.append((lvl, tol, its, ref, path))

    if not bad:
        print(f"{a.study}: no anomalies (factor {a.factor})")
        return 0

    print(f"{a.study}: {len(bad)} anomalous cell(s) at factor {a.factor}")
    print(f"{'cell':10s} {'iters':>7s} {'nbr':>8s} {'ratio':>6s}")
    for lvl, tol, its, ref, path in bad:
        print(f"L{lvl}/{tol:<7s} {its:7.1f} {ref:8.1f} {its/ref:5.2f}x")
        if a.delete:
            os.remove(path)

    lv = " ".join(str(l) for l in sorted({b[0] for b in bad}))
    tl = " ".join(sorted({b[1] for b in bad}, key=float, reverse=True))
    print(f"\n{'DELETED — ' if a.delete else 'dry run; add --delete to remove. '}re-run with:")
    print(f'  LEVELS="{lv}" TOLS="{tl}"')
    print("  (note: this re-runs the full LEVELS x TOLS box; cells whose logs still exist are skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
