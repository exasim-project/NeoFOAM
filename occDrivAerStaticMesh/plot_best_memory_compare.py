#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
"""Cross-config memory comparison for a param-study-best.sh run.

param-study-best.sh runs three pressure-solver configs (fp64 localized-MG, its float counterpart, and
a PCG+diagonal baseline) and drops, per run, a NF_MEM_SCOPE device-pool CSV (<stem>.memoryTimeline.csv)
and an Umpire allocation-replay CSV (<stem>.replay-foot.csv). This script overlays all three so the
memory footprint can be compared across solver choices, and contrasts the two measurement paths:

  * Panel 1 -- the NF_MEM_SCOPE "live (current)" pool timeline for each config (rank-representative),
    showing the per-timestep sawtooth the scoped probes DO see.
  * Panel 2 -- grouped bars per config: the CSV live-peak (what the scoped probes report) vs the
    Umpire-replay true peak (max simultaneously-live across all ranks, transients included) vs the
    reserved pool. The gap between the first two is the transient headroom the CSV misses.

Usage:
    python3 plot_best_memory_compare.py [RESULTS_DIR] [-o out.png] [-n CELLS_PER_RANK]
    # RESULTS_DIR defaults to paramStudyResults/best-practice
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MB = 1 << 20


def short_label(stem):
    """Human label from a run stem (strip the trailing -YYYYmmdd-HHMMSS timestamp)."""
    base = os.path.basename(stem)
    if "pMG-localized-precfloat" in base:
        return "float MG (precfloat)"
    if "pMG-localized-best" in base:
        return "fp64 MG (localized+cache)"
    if "pPCG-diagonal" in base:
        return "PCG + diagonal"
    return base.rsplit("-", 2)[0]


def load_timeline(path):
    """seq + live-current MB + reserved MB from a memoryTimeline.csv."""
    seq, cur, act = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            seq.append(int(r["seq"]))
            cur.append(int(r["current_bytes"]) / MB)
            act.append(int(r["actual_bytes"]) / MB)
    return seq, cur, act


def replay_peak_mb(path):
    """Max simultaneously-live MB across ALL ranks in a replay-foot.csv.

    The CSV has one row per allocation event: (event, t_ms, allocator_ref, allocator, live_bytes),
    where live_bytes is that allocator's running live total AFTER the event. Peak per allocator =
    max over its rows; total device peak ~ sum of per-allocator peaks (ranks peak near-simultaneously
    at the per-timestep assembly spike, so the sum is the right whole-GPU-set figure). We report both
    the per-rank max and the summed peak."""
    per_alloc = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            a = r["allocator"]
            v = int(r["live_bytes"]) / MB
            if v > per_alloc.get(a, 0):
                per_alloc[a] = v
    if not per_alloc:
        return 0.0, 0.0
    return max(per_alloc.values()), sum(per_alloc.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="paramStudyResults/best-practice")
    ap.add_argument("-o", "--out", default="paramStudyResults/best-practice/memory-compare.png")
    ap.add_argument("-n", "--cells", type=float, default=None,
                    help="per-rank cell count -> adds a bytes/cell axis on panel 1")
    args = ap.parse_args()

    tls = sorted(glob.glob(os.path.join(args.results_dir, "*.memoryTimeline.csv")))
    if not tls:
        sys.exit(f"no *.memoryTimeline.csv in {args.results_dir}")

    # Keep only the NEWEST run per config label (the trailing -YYYYmmdd-HHMMSS sorts lexically), so a
    # stale timeline from an earlier invocation in the same dir does not duplicate a config.
    newest = {}
    for tl in tls:
        stem = tl[: -len(".memoryTimeline.csv")]
        lab = short_label(stem)
        if lab not in newest or stem > newest[lab]:
            newest[lab] = stem
    configs = []
    for lab, stem in sorted(newest.items(), key=lambda kv: kv[1]):
        rp = stem + ".replay-foot.csv"
        configs.append(dict(label=lab, tl=stem + ".memoryTimeline.csv",
                            rp=rp if os.path.exists(rp) else None))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw=dict(width_ratios=[3, 2]))
    colors = ["#249", "#c44", "#2a2", "#a5a"]

    # ---- Panel 1: overlaid live-current timelines ------------------------------------------------
    peak_reserved = 0
    for i, c in enumerate(configs):
        seq, cur, act = load_timeline(c["tl"])
        c["csv_live_peak"] = max(cur) if cur else 0
        c["csv_reserved"] = max(act) if act else 0
        peak_reserved = max(peak_reserved, c["csv_reserved"])
        ax0.plot(seq, cur, lw=1.0, color=colors[i % len(colors)], label=c["label"])
    ax0.set_xlabel("probe sequence")
    ax0.set_ylabel("device pool live [MB]")
    ax0.set_title("NF_MEM_SCOPE live footprint (per-config, rank-representative)")
    ax0.legend(loc="lower right", fontsize=8)
    ax0.grid(alpha=0.2)
    if args.cells:
        secax = ax0.secondary_yaxis(
            "right", functions=(lambda m: m * MB / args.cells, lambda b: b * args.cells / MB))
        secax.set_ylabel(f"bytes per cell ({args.cells:,.0f} cells)")

    # ---- Panel 2: CSV-peak vs replay-true-peak vs reserved ---------------------------------------
    for c in configs:
        if c["rp"]:
            c["replay_rank_peak"], c["replay_sum_peak"] = replay_peak_mb(c["rp"])
        else:
            c["replay_rank_peak"] = c["replay_sum_peak"] = 0

    labels = [c["label"] for c in configs]
    x = range(len(configs))
    w = 0.26
    csv_live = [c["csv_live_peak"] for c in configs]
    rep_rank = [c["replay_rank_peak"] for c in configs]
    reserved = [c["csv_reserved"] for c in configs]
    ax1.bar([xi - w for xi in x], csv_live, w, color="#249", label="CSV live peak (scoped)")
    ax1.bar([xi for xi in x], rep_rank, w, color="#c44", label="replay true peak (per rank)")
    ax1.bar([xi + w for xi in x], reserved, w, color="#bbb", label="reserved pool")
    # Annotate the transient gap (replay - csv) on each config.
    for xi, c in zip(x, configs):
        gap = c["replay_rank_peak"] - c["csv_live_peak"]
        if gap > 0:
            ax1.annotate(f"+{gap:,.0f}\ntransient", (xi, c["replay_rank_peak"]),
                         textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7,
                         color="#c44")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8, rotation=12, ha="right")
    ax1.set_ylabel("device memory [MB]")
    ax1.set_title("Peak footprint: scoped-CSV vs allocation-replay vs reserved")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}\n")

    print(f"{'config':<28} {'CSV live':>10} {'replay/rank':>12} {'replay sum':>11} {'reserved':>10}")
    for c in configs:
        print(f"{c['label']:<28} {c['csv_live_peak']:>10,.0f} {c['replay_rank_peak']:>12,.0f} "
              f"{c['replay_sum_peak']:>11,.0f} {c['csv_reserved']:>10,.0f}   [MB]")


if __name__ == "__main__":
    main()
