#!/usr/bin/env python
"""Render the feature-coverage view from a call_graph.py ``callgraph.json`` as a
single image: the most-central public functions, colored by the highest test
level that reaches each — so "important feature with no integration test" reads
red at a glance.

    python .claude/skills/understand-tests/feature_chart.py \
        --json <out>/callgraph.json --out <out>/feature_coverage.png [--top 30]

x = in-degree (how many source calls depend on the fn = centrality ≈ importance).
color = best covering test level: integration (green) · component (teal) ·
unit (amber) · GAP (red, no test statically reaches it). Needs matplotlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Static mode: level -> (color, label).
LEVEL_STYLE = {
    "integration": ("#2e7d32", "integration (e2e)"),
    "component": ("#00838f", "component"),
    "unit": ("#f9a825", "unit only"),
    "GAP": ("#c62828", "GAP — no test reaches it"),
}
# Dynamic mode: per-fn coverage verdict -> (color, label).
DYN_STYLE = {
    "covered": ("#2e7d32", "fully covered (all lines ran)"),
    "partial": ("#f9a825", "partially covered"),
    "uncovered": ("#c62828", "uncovered (no line ran)"),
    "unknown": ("#9e9e9e", "no coverage data for file"),
}


def _dynamic_verdicts(cov: dict, defs: list[dict]) -> dict[str, str]:
    """Map each source fn qualname -> covered/partial/uncovered by intersecting
    the coverage file's missing lines with the fn's [lineno, end_lineno] span."""
    # coverage file path (may be absolute/site-packages) -> set(missing lines)
    miss_by_suffix: dict[str, set[int]] = {}
    for f in cov.get("files", []):
        # index by a trailing path chunk so worktree/site-packages paths still join
        key = "/".join(Path(f["file"]).parts[-3:])
        miss_by_suffix[key] = set(f.get("missing_lines", []))
    out: dict[str, str] = {}
    for d in defs:
        key = "/".join(Path(d["file"]).parts[-3:])
        if key not in miss_by_suffix:
            out[d["qualname"]] = "unknown"
            continue
        # Body span excludes the `def` line, which coverage marks "run" at import
        # even for a never-called method — counting it would inflate to "partial".
        body = set(range(d["lineno"] + 1, d.get("end_lineno", d["lineno"]) + 1))
        if not body:
            out[d["qualname"]] = "covered"
            continue
        missing_frac = len(body & miss_by_suffix[key]) / len(body)
        out[d["qualname"]] = (
            "covered"
            if missing_frac == 0
            else "uncovered"
            if missing_frac >= 0.7
            else "partial"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, help="Path to callgraph.json.")
    ap.add_argument("--out", required=True, help="Output image path (.png/.svg).")
    ap.add_argument("--top", type=int, default=30, help="Show the top-N central fns.")
    ap.add_argument("--title", default=None, help="Override the chart title.")
    ap.add_argument(
        "--coverage-json",
        default=None,
        help="Optional coverage_contexts.py coverage.json. When given, "
        "bars are colored by DYNAMIC per-fn coverage (covered / "
        "partial / uncovered) instead of static test level — i.e. "
        "'green = lines actually ran', not 'a test touches it'.",
    )
    args = ap.parse_args()

    report = json.loads(Path(args.json).read_text())
    feats = report.get("feature_coverage")
    if not feats:
        print(
            "No feature_coverage in JSON (run call_graph.py with --tests).",
            file=sys.stderr,
        )
        return 2

    dyn = None
    if args.coverage_json:
        cov = json.loads(Path(args.coverage_json).read_text())
        dyn = _dynamic_verdicts(cov, report.get("definitions", []))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed.", file=sys.stderr)
        return 2

    # Top-N by in-degree; keep GAPs even at in-degree 0 if they crack the top by
    # centrality. Reverse so the biggest bar is at the top of a horizontal chart.
    def _short(qual: str) -> str:
        # neofoam.a.b.mod:Class.method -> mod:Class.method
        mod, _, sym = qual.partition(":")
        return f"{mod.split('.')[-1]}:{sym}" if sym else mod.split(".")[-1]

    rows = feats[: args.top][::-1]
    names = [_short(r["fn"]) for r in rows]
    vals = [
        max(r["in_degree"], 0.4) for r in rows
    ]  # floor so 0-in-degree GAPs are visible
    if dyn is not None:
        keys = [dyn.get(r["fn"], "unknown") for r in rows]
        colors = [DYN_STYLE[k][0] for k in keys]
        style, present = DYN_STYLE, set(keys)
    else:
        keys = [r["best_test_level"] for r in rows]
        colors = [LEVEL_STYLE.get(k, ("#9e9e9e", "?"))[0] for k in keys]
        style, present = LEVEL_STYLE, set(keys)

    s = report["summary"]
    hist = s.get("test_level_histogram", {})
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, max(4, n * 0.34)))
    ax.barh(range(n), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8, fontfamily="monospace")
    ax.set_xlabel(
        "in-degree  (source calls depending on this fn  ≈  centrality / importance)",
        fontsize=9,
    )
    default_title = f"Feature coverage — {s.get('scope') or 'module'}"
    ax.set_title(args.title or default_title, fontsize=13, fontweight="bold", pad=14)

    # subtitle line with the headline numbers
    if dyn is not None:
        known = [k for k in dyn.values() if k != "unknown"]
        n_cov = sum(1 for k in known if k == "covered")
        sub = (
            f"colored by DYNAMIC coverage — {n_cov}/{len(known)} covered source fns "
            f"fully run   |   tests: " + " · ".join(f"{k} {v}" for k, v in hist.items())
        )
    else:
        sub = (
            f"{s.get('public_api_fns', '?')} public fns · "
            f"{s.get('public_api_static_gaps', '?')} with NO test reaching them   |   "
            f"tests: " + " · ".join(f"{k} {v}" for k, v in hist.items())
        )
    ax.text(
        0, 1.005, sub, transform=ax.transAxes, fontsize=8.5, color="#444", va="bottom"
    )

    handles = [
        mpatches.Patch(color=c, label=lbl)
        for k, (c, lbl) in style.items()
        if k in present
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    ax.margins(y=0.01)
    footer = (
        "Bars colored by measured line coverage (lines that actually ran). Coverage proves execution, not assertion strength."
        if dyn is not None
        else "Static best-effort. Altitude = touch surface, not assertion focus; GAP is static — confirm against the dynamic coverage pass."
    )
    fig.text(0.01, 0.005, footer, fontsize=7, color="#888")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
