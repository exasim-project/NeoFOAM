#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
"""Condense a MemoryProbe allocation-records dump (<stem>.allocRecords.txt) into a field-level ranking.

The dump (written by MemoryProbe::dumpAllocatorRecords when NEOFOAM_MEM_ALLOC_RECORDS=1 +
UMPIRE_BACKTRACE=On) lists the peak's live blocks, each with the full 17-frame allocating call stack.
That is precise but verbose. This script folds each block down to two things:

  * allocated type  -- the NeoN::Vector<T> being constructed (what the bytes ARE), and
  * semantic site   -- the first application frame above the allocator/Vector plumbing (what the
                       bytes are FOR: UnstructuredMesh, finiteVolume::cellCentred field, la::LinearSystem,
                       TurbulenceModel, PDE assemble, ...), plus one caller frame for context.

then re-aggregates by (type, site) so the output reads as a ranked list of named fields rather than
raw stacks.

Usage:
    python3 summarize_alloc_records.py [FILE_OR_DIR ...] [--csv out.csv] [--top N]
    # no args -> globs paramStudyResults/best-practice/*.allocRecords.txt
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

# Frames that are allocator/runtime plumbing, not a meaningful "what is this memory for" site.
PLUMBING = re.compile(
    r"umpire::|No dladdr|UmpirePoolAllocator::alloc|operator new|__gnu_cxx|"
    r"std::|__libc|_start\b|::allocate\b|AllocatorStrategy|Inspector::registerAllocation"
)
# The Vector<T> constructor is the allocated type, captured separately (not used as the site).
VECTOR_CTOR = re.compile(r"NeoN::Vector<.*>::Vector")
FRAME = re.compile(r"^\s*[0-9a-f]+\s+0x[0-9a-f]+\s+(.*?)(?:\+0x[0-9a-f]+)?(?:\s+\[0x[0-9a-f]+\])?\s*$")
HEAD = re.compile(r"^(\d+)\s*MB\s+x(\d+)\s+\(largest\s+(\d+)\s*MB\)")


def sym(frame_symbol):
    """Trim a demangled symbol to Class::method, keeping template args on the type but dropping the
    argument list and the trailing +offset."""
    s = frame_symbol.strip()
    # cut the function argument list: first '(' at paren-depth following the last template '>' close.
    depth = 0
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            return s[:i]
    return s


def parse_blocks(path):
    """Yield (mb, count, largest, [frame_symbols]) per block in an allocRecords.txt."""
    with open(path) as f:
        lines = f.readlines()
    i, n = 0, len(lines)
    while i < n:
        m = HEAD.match(lines[i])
        if not m:
            i += 1
            continue
        mb, count, largest = int(m.group(1)), int(m.group(2)), int(m.group(3))
        i += 1
        frames = []
        while i < n and not HEAD.match(lines[i]):
            fm = FRAME.match(lines[i])
            if fm and "Backtrace:" not in lines[i]:
                frames.append(fm.group(1).strip())
            i += 1
        yield mb, count, largest, frames


def classify(frames):
    """From a block's frames pick (allocated_type, site, context)."""
    alloc_type = None
    for fr in frames:
        if VECTOR_CTOR.search(fr):
            alloc_type = sym(fr)
            break
    # Semantic frames = application frames that are not plumbing and not the Vector ctor / main tail.
    semantic = [sym(fr) for fr in frames
                if not PLUMBING.search(fr) and not VECTOR_CTOR.search(fr) and "main" != sym(fr)]
    site = semantic[0] if semantic else (alloc_type or "?")
    context = semantic[1] if len(semantic) > 1 else ""
    return alloc_type or "?", site, context


def summarize(path, top, cells=None):
    agg = defaultdict(lambda: [0, 0])  # (type, site, context) -> [mb, count]
    total = 0
    trigger, snapshot = "?", "?"
    with open(path) as f:
        for ln in f:
            if ln.startswith("# trigger tag"):
                trigger = ln.split(":", 1)[1].strip()
            elif ln.startswith("# live at snapshot"):
                snapshot = ln.split(":", 1)[1].strip()
    for mb, count, _largest, frames in parse_blocks(path):
        t, site, ctx = classify(frames)
        key = (t, site, ctx)
        agg[key][0] += mb
        agg[key][1] += count
        total += mb
    ranked = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)

    snap_mb = None
    m = re.match(r"(\d+)\s*MB", snapshot)
    if m:
        snap_mb = int(m.group(1))
    cov = f"; listed blocks cover {total:,} of {snap_mb:,} MB" if snap_mb else ""
    print(f"\n#### {os.path.basename(path)}")
    print(f"     peak snapshot {snapshot} at tag '{trigger}'{cov}")

    # bytes-per-cell helper (per-rank cell count); the region MB IS its total footprint.
    def bpc(mb):
        return f"{mb * (1 << 20) / cells:>8.1f}" if cells else ""

    hdr_bpc = f" {'B/cell':>8}" if cells else ""
    print(f"     {'MB':>7} {'x':>5}{hdr_bpc}  allocated type  <=  semantic site  <=  context")
    for (t, site, ctx), (mb, count) in ranked[:top]:
        tail = f"  <=  {ctx}" if ctx else ""
        print(f"     {mb:>7,} {count:>5}{(' ' + bpc(mb)) if cells else ''}  {t}  <=  {site}{tail}")

    # Totals: the listed regions, plus (if known) the full snapshot.
    shown_mb = sum(v[0] for (_k, v) in ranked[:top])
    shown_ct = sum(v[1] for (_k, v) in ranked[:top])
    print(f"     {'-' * 7} {'-' * 5}{(' ' + '-' * 8) if cells else ''}")
    print(f"     {shown_mb:>7,} {shown_ct:>5}{(' ' + bpc(shown_mb)) if cells else ''}  "
          f"TOTAL of top-{min(top, len(ranked))} listed regions")
    if snap_mb:
        print(f"     {snap_mb:>7,} {'':>5}{(' ' + bpc(snap_mb)) if cells else ''}  "
              f"FULL peak snapshot (incl. un-listed tail)")
    return path, total, ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*",
                    help="allocRecords.txt files or dirs; default paramStudyResults/best-practice")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("-n", "--cells", type=float, default=None,
                    help="per-rank cell count -> adds a bytes-per-cell column and a TOTAL row")
    ap.add_argument("--csv", help="write the (file,type,site,context,mb,count) rows to CSV")
    args = ap.parse_args()

    paths = []
    for x in args.inputs or ["paramStudyResults/best-practice"]:
        if os.path.isdir(x):
            paths += sorted(glob.glob(os.path.join(x, "*.allocRecords.txt")))
        elif any(c in x for c in "*?["):
            paths += sorted(glob.glob(x))
        else:
            paths.append(x)
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit("no *.allocRecords.txt found (run with NEOFOAM_MEM_ALLOC_RECORDS=1 + UMPIRE_BACKTRACE=On)")

    rows = []
    for p in paths:
        _, _, ranked = summarize(p, args.top, args.cells)
        for (t, site, ctx), (mb, count) in ranked:
            rows.append((os.path.basename(p), t, site, ctx, mb, count))

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "allocated_type", "site", "context", "mb", "count"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
