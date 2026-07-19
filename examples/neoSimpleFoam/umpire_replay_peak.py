#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
"""Reduce an Umpire *replay* event log to a true, allocation-granularity memory footprint.

The NF_MEM_SCOPE CSV timeline (see memoryProbe.hpp / plot_memory_timeline.py) only samples the
device pool at the annotation points you place, so a temporary that is allocated and freed *between*
two probes is invisible. Umpire's own event log does not have that blind spot: with

    UMPIRE_REPLAY=1  ./neoSimpleFoam ...          # (UMPIRE_EVENTS=1 works too)
    # optional: UMPIRE_OUTPUT_DIR=<dir> UMPIRE_OUTPUT_BASENAME=<name>

Umpire writes one JSON object *per line* (JSONL) recording every allocate / named_allocate /
deallocate with its size, pointer, owning allocator and a nanosecond timestamp, to a file named
    <UMPIRE_OUTPUT_DIR>/<basename>.<pid>.<n>.stats            (default ./umpire.<pid>.0.stats)

This script replays those events in chronological order, reconstructs the live-bytes curve for each
allocator, and reports:

  * the PEAK live footprint per allocator (the OOM-relevant number, transients included),
  * how much of that peak is transient headroom (peak - final live), i.e. scratch that was released,
  * the composition of the peak (the allocations that were live at the peak instant), attributed by
    name when Umpire recorded named_allocate, otherwise bucketed by size, and
  * the largest short-lived allocations (the transient temporaries themselves).

Optionally it writes the reconstructed footprint timeline to CSV (--csv) and/or a PNG (--plot),
which can be laid next to the NF_MEM_SCOPE timeline to see exactly what the scoped probes missed.

Usage:
    python3 umpire_replay_peak.py [umpire.*.stats ...] [--csv foot.csv] [--plot foot.png]
    # with no path it globs ./umpire.*.stats and ./*.stats
"""

import argparse
import glob
import json
import sys
from collections import defaultdict

MB = 1 << 20


def find_inputs(paths):
    if paths:
        out = []
        for p in paths:
            out.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
        return out
    # default: whatever Umpire dropped in the cwd
    found = sorted(set(glob.glob("umpire.*.stats") + glob.glob("*.stats")))
    return found


def parse_events(path):
    """Yield (name, size, ref, ptr, alloc_name, t_ns) for the allocate/deallocate events.

    Umpire writes malformed-JSON-tolerant JSONL; skip anything that does not parse or is not one of
    the memory operations we care about."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = e.get("name")
            if name not in ("allocate", "named_allocate", "deallocate"):
                continue
            sa = e.get("string_args", {})
            na = e.get("numeric_args", {})
            yield (
                name,
                int(na.get("size", 0)),
                sa.get("allocator_ref", "?"),
                sa.get("pointer", "?"),
                sa.get("allocation_name"),
                int(e.get("timestamp", 0)),
            )


class AllocState:
    """Per-allocator live-bytes bookkeeping."""

    def __init__(self):
        self.live = {}          # ptr -> (size, name, t_alloc)
        self.total = 0          # current live bytes
        self.peak = 0           # max live bytes ever
        self.peak_t = 0         # timestamp of the peak
        self.peak_snapshot = [] # (size, name) live at the peak instant
        self.n_alloc = 0
        self.largest = 0        # largest single allocation
        self.lifetimes = []     # (size, name, dt_ns) for freed allocations -> transients


def replay(events):
    """Chronologically fold events into per-allocator AllocState, and build a global footprint
    timeline (relative-ms, per-allocator live MB) for plotting."""
    st = defaultdict(AllocState)
    timeline = []  # (t_ns, ref, total_bytes)
    t0 = None
    for name, size, ref, ptr, aname, t in events:
        if t0 is None:
            t0 = t
        s = st[ref]
        if name in ("allocate", "named_allocate"):
            s.live[ptr] = (size, aname, t)
            s.total += size
            s.n_alloc += 1
            s.largest = max(s.largest, size)
            if s.total > s.peak:
                s.peak = s.total
                s.peak_t = t
                s.peak_snapshot = [(sz, nm) for sz, nm, _ in s.live.values()]
        else:  # deallocate
            rec = s.live.pop(ptr, None)
            if rec is not None:
                sz, nm, ta = rec
                s.total -= sz
                s.lifetimes.append((sz, nm, t - ta))
        timeline.append((t, ref, s.total))
    return st, timeline, (t0 or 0)


def label_allocators(st):
    """Map each hex allocator ref to a short stable label A0, A1, ... in peak order."""
    order = sorted(st, key=lambda r: st[r].peak, reverse=True)
    return {ref: f"A{i}" for i, ref in enumerate(order)}, order


def human(nbytes_):
    return f"{nbytes_ / MB:,.1f} MB" if nbytes_ else "0"


def report(st, order, labels):
    print(f"{len(st)} allocator(s) recorded. Ranked by peak live footprint:\n")
    for ref in order:
        s = st[ref]
        if s.n_alloc == 0:
            continue
        lab = labels[ref]
        transient = s.peak - s.total
        print(f"[{lab}] ref={ref}")
        print(f"    allocations         : {s.n_alloc:,}")
        print(f"    peak live footprint : {human(s.peak)}")
        print(f"    final live (leak)   : {human(s.total)}")
        print(f"    transient headroom  : {human(transient)}   (peak - final; scratch that was freed)")
        print(f"    largest single alloc: {human(s.largest)}")

        # Composition of the peak: attribute by name when available, else bucket by size.
        named = defaultdict(lambda: [0, 0])  # key -> [bytes, count]
        any_named = False
        for sz, nm in s.peak_snapshot:
            key = nm if nm else f"<anon {human(sz)}>"
            if nm:
                any_named = True
            named[key][0] += sz
            named[key][1] += 1
        top = sorted(named.items(), key=lambda kv: kv[1][0], reverse=True)[:8]
        head = "named fields live at peak" if any_named else "largest live blocks at peak (anon)"
        print(f"    {head}:")
        for key, (b, c) in top:
            print(f"        {human(b):>12}  x{c:<5} {key}")

        # Transient temporaries: largest short-lived allocations (freed, ranked by size).
        freed = sorted((x for x in s.lifetimes if x[0] > 0), key=lambda x: x[0], reverse=True)[:6]
        if freed:
            print("    largest freed (transient) allocations:")
            for sz, nm, dt in freed:
                tag = nm if nm else "<anon>"
                print(f"        {human(sz):>12}  lived {dt / 1e6:8.2f} ms   {tag}")
        print()


def write_csv(path, timeline, t0, labels):
    with open(path, "w") as f:
        f.write("event,t_ms,allocator_ref,allocator,live_bytes\n")
        for i, (t, ref, total) in enumerate(timeline):
            f.write(f"{i},{(t - t0) / 1e6:.6f},{ref},{labels.get(ref, '?')},{total}\n")
    print(f"wrote {path}  ({len(timeline)} events)")


def plot(path, timeline, t0, st, order, labels, topk=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = order[:topk]
    series = {ref: ([], []) for ref in keep}
    for t, ref, total in timeline:
        if ref in series:
            series[ref][0].append((t - t0) / 1e6)
            series[ref][1].append(total / MB)

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#249", "#c44", "#2a2", "#a5a", "#888"]
    for i, ref in enumerate(keep):
        xs, ys = series[ref]
        s = st[ref]
        ax.plot(xs, ys, lw=1.0, color=colors[i % len(colors)], drawstyle="steps-post",
                label=f"{labels[ref]}  peak {s.peak / MB:,.0f} MB, {s.n_alloc:,} allocs")
        ax.axhline(s.peak / MB, color=colors[i % len(colors)], lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("live device bytes [MB]")
    ax.set_title("Umpire replay: allocation-granularity live footprint (dotted = per-allocator peak)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats", nargs="*", help="Umpire .stats JSONL file(s); default globs ./*.stats")
    ap.add_argument("--csv", help="write reconstructed footprint timeline to this CSV")
    ap.add_argument("--plot", help="write a footprint-over-time PNG")
    args = ap.parse_args()

    inputs = find_inputs(args.stats)
    if not inputs:
        sys.exit("no Umpire .stats files found (run with UMPIRE_REPLAY=1 and check UMPIRE_OUTPUT_DIR)")
    print(f"reading {len(inputs)} file(s): {', '.join(inputs)}\n")

    events = []
    for p in inputs:
        events.extend(parse_events(p))
    if not events:
        sys.exit("no allocate/deallocate events parsed (is this an Umpire replay/event log?)")
    # Sort by timestamp so multiple ranks / files interleave correctly.
    events.sort(key=lambda e: e[5])

    st, timeline, t0 = replay(events)
    labels, order = label_allocators(st)
    report(st, order, labels)

    if args.csv:
        write_csv(args.csv, timeline, t0, labels)
    if args.plot:
        plot(args.plot, timeline, t0, st, order, labels)


if __name__ == "__main__":
    main()
