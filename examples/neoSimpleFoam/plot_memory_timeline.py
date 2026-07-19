#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
"""Plot the NeoFOAM memory-footprint timeline.

Reads the CSV written by MemoryProbe::dump() (enable at runtime with
NEOFOAM_MEM_TIMELINE=1; see include/NeoFOAM/auxiliary/memoryProbe.hpp) and produces:

  1. a timeline of device-pool footprint (current / actual-reserved / high-water) vs. probe
     sequence, with per-timestep boundaries marked, and
  2. a "bottleneck" bar chart ranking the scoped regions (NF_MEM_SCOPE enter/exit pairs) by the
     current-bytes they hold live, so the region driving the footprint is obvious.

Usage:
    python3 plot_memory_timeline.py [memoryTimeline.csv] [-o out.png]
"""

import argparse
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MB = 1 << 20


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                dict(
                    seq=int(r["seq"]),
                    step=int(r["step"]),
                    time=float(r["time"]),
                    tag=r["tag"],
                    phase=r["phase"],
                    current=int(r["current_bytes"]),
                    actual=int(r["actual_bytes"]),
                    highwater=int(r["highwater_bytes"]),
                )
            )
    rows.sort(key=lambda x: x["seq"])
    return rows


def scope_deltas(rows):
    """Match enter/exit rows per tag (stack-based, supports nesting) and return
    {tag: [current-delta bytes, ...]} — the net footprint each scoped region held live."""
    stacks = defaultdict(list)
    deltas = defaultdict(list)
    for r in rows:
        if r["phase"] == "enter":
            stacks[r["tag"]].append(r["current"])
        elif r["phase"] == "exit" and stacks[r["tag"]]:
            deltas[r["tag"]].append(r["current"] - stacks[r["tag"]].pop())
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="memoryTimeline.csv")
    ap.add_argument("-o", "--out", default="memoryTimeline.png")
    ap.add_argument(
        "-n", "--cells", type=float, default=None,
        help="number of cells this timeline covers (e.g. per-rank cell count); "
        "adds a right-hand axis showing footprint in bytes per cell",
    )
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        sys.exit(f"no samples in {args.csv} (was NEOFOAM_MEM_TIMELINE=1 set?)")

    seq = [r["seq"] for r in rows]
    cur = [r["current"] / MB for r in rows]
    act = [r["actual"] / MB for r in rows]
    high = [r["highwater"] / MB for r in rows]

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(13, 9), gridspec_kw=dict(height_ratios=[2, 1])
    )

    # ---- Panel 1: footprint timeline -------------------------------------------------
    ax0.plot(seq, act, label="reserved (actual)", color="#c44", lw=1.2)
    ax0.plot(seq, cur, label="live (current)", color="#249", lw=1.2)
    ax0.plot(seq, high, label="high-water", color="#888", lw=1.0, ls="--")
    ax0.fill_between(seq, cur, color="#249", alpha=0.12)

    # Mark per-step boundaries (first sample of each new step index >= 0).
    seen = set()
    for r in rows:
        if r["step"] >= 0 and r["step"] not in seen:
            seen.add(r["step"])
            ax0.axvline(r["seq"], color="k", alpha=0.08, lw=0.8)
            ax0.text(
                r["seq"], ax0.get_ylim()[1], f" {r['step']}",
                fontsize=6, color="k", alpha=0.5, va="top",
            )

    ax0.set_xlabel("probe sequence")
    ax0.set_ylabel("device pool [MB]")
    ax0.set_title("NeoFOAM device-memory footprint timeline (thin vlines = timestep starts)")
    ax0.legend(loc="upper left", fontsize=9)
    ax0.grid(alpha=0.2)

    # Optional right-hand axis: same data expressed as bytes per cell. The MB axis value m maps
    # to m * MB / cells bytes-per-cell, so the two axes stay locked as the view is rescaled.
    if args.cells:
        secax = ax0.secondary_yaxis(
            "right",
            functions=(lambda m: m * MB / args.cells, lambda b: b * args.cells / MB),
        )
        secax.set_ylabel(f"bytes per cell  ({args.cells:,.0f} cells)")

    # ---- Panel 2: per-region bottleneck ranking --------------------------------------
    deltas = scope_deltas(rows)
    # Rank by the max live footprint any single invocation of the region held.
    ranked = sorted(
        ((t, max(d) / MB, sum(d) / len(d) / MB) for t, d in deltas.items() if d),
        key=lambda x: x[1],
        reverse=True,
    )
    if ranked:
        tags = [t for t, _, _ in ranked]
        peak = [p for _, p, _ in ranked]
        mean = [m for _, _, m in ranked]
        y = range(len(tags))
        ax1.barh(y, peak, color="#c44", alpha=0.55, label="peak Δlive")
        ax1.barh(y, mean, color="#249", alpha=0.85, label="mean Δlive")
        ax1.set_yticks(list(y))
        ax1.set_yticklabels(tags, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("net live footprint held by region [MB]")
        ax1.set_title("Footprint by scoped region (NF_MEM_SCOPE enter/exit delta) — bottlenecks on top")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(alpha=0.2, axis="x")
    else:
        ax1.text(0.5, 0.5, "no NF_MEM_SCOPE regions recorded", ha="center")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({len(rows)} samples, {len(deltas)} scoped regions)")

    # Also print a terminal summary of the top regions.
    if ranked:
        print("\nTop memory regions (net live footprint):")
        print(f"  {'region':<32} {'peak MB':>10} {'mean MB':>10}")
        for t, p, m in ranked[:15]:
            print(f"  {t:<32} {p:>10.1f} {m:>10.1f}")


if __name__ == "__main__":
    main()
