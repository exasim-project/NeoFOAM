#!/usr/bin/env python
"""Per-test line-coverage + redundancy analysis for the spec-loop test reviewer.

Project-agnostic Python coverage helper. Some projects ship **native extension
modules** (pybind11 / Cython / a compiled `.so`) that abort the process when they
are imported while coverage's tracer is armed — which makes the standard
``pytest --cov`` plugin unusable. This script works around that: it imports any
modules named via ``--preimport`` (and the feature itself) **uninstrumented first**,
then arms coverage (scoped to the feature package) with per-test *dynamic contexts*,
then runs the feature's tests in-process. For projects with no native modules,
``--preimport`` is simply unused and it behaves like an ordinary scoped coverage run.

It answers "which test covered which line", which drives safe test
simplification: tests that cover an identical (or subset) line set are
merge/parametrize candidates, while a line owned by a single test marks that test
as load-bearing (never drop it).

Generic — nothing here is feature-specific. Pass any package(s) and test path(s):

    python .claude/agents/coverage_contexts.py \
        --source my_pkg.my_feature \
        --tests  test/my_feature/ \
        --out    loop/my_feature/iter-1/coverage
        # add: --preimport my_native_ext   (only if importing aborts under coverage)

Outputs (under --out):
  * coverage.json  — summary, per-test line counts, identical/subset clusters,
                     sole-owner lines, missing lines (machine-readable handoff)
  * coverage.md    — readable summary + a Mermaid cluster/subset graph the LLM
                     reviewer can embed straight into 4-test-review.md
  * clusters.mmd   — the Mermaid graph source on its own
  * htmlcov/       — HTML report with show_contexts (hover a line -> tests)
  * similarity_heatmap.png — only with --viz; Jaccard heatmap (needs matplotlib)
Every similarity artifact carries the "coverage-shape only — confirm assertions
before merging" guardrail in its caption. Prints a concise human summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _split(values: list[str]) -> list[str]:
    """Allow both repeated flags and comma-separated values."""
    out: list[str] = []
    for v in values or []:
        out.extend(p.strip() for p in v.split(",") if p.strip())
    return out


def _caption(discriminates: bool) -> str:
    """The guardrail that MUST travel with every similarity visualization."""
    base = (
        "Coverage-shape similarity only: tests grouped here execute identical "
        "lines, which does NOT imply identical assertions. Confirm the "
        "assertions before merging."
    )
    if not discriminates:
        base += (
            " line_coverage_discriminates=false — no test sole-owns a line, "
            "so coverage alone cannot prove redundancy in this module."
        )
    return base


def _mermaid(
    identical_clusters: list,
    subset_pairs: list[dict],
    singletons: list[str],
    caption: str,
    cap: int = 10,
) -> str:
    """Render the test-to-test cluster/subset structure as Mermaid (text, the
    format the LLM reviewer can actually read and embed)."""
    lines = ["flowchart TD"]
    for part in (caption[i : i + 90] for i in range(0, len(caption), 90)):
        lines.append(f"    %% {part}")
    ids: dict[str, str] = {}

    def nid(name: str) -> str:
        return ids.setdefault(name, f"t{len(ids)}")

    for i, (group, nlines) in enumerate(identical_clusters):
        shown = group[:cap]
        lines.append(
            f'    subgraph C{i}["identical: {len(group)} tests · '
            f'{nlines} lines → parametrize candidate"]'
        )
        for t in shown:
            lines.append(f'        {nid(t)}["{t}"]')
        if len(group) > cap:
            lines.append(f'        C{i}more["… +{len(group) - cap} more"]')
        lines.append("    end")
    for t in singletons[: cap * 3]:
        lines.append(f'    {nid(t)}["{t}"]')
    # subset edges (a is fully contained in b w.r.t. lines)
    seen = 0
    for p in subset_pairs:
        a, b = p["subset"], p["superset"]
        if a in ids and b in ids:
            lines.append(f"    {ids[a]} -. subset .-> {ids[b]}")
            seen += 1
            if seen >= 40:
                break
    return "\n".join(lines)


def _markdown(report: dict, mermaid: str, caption: str) -> str:
    """A small markdown doc the reviewer can read and embed into 4-test-review."""
    out = [
        f"# Coverage contexts — {', '.join(report['sources'])}",
        "",
        f"**Line coverage:** {report['total_percent']}% "
        f"({report['total_missing']}/{report['total_statements']} missing) · "
        f"{report['num_tests']} tests · "
        f"`line_coverage_discriminates={str(report['line_coverage_discriminates']).lower()}`",
        "",
        f"> {caption}",
        "",
        "## Test-similarity structure",
        "",
        "```mermaid",
        mermaid,
        "```",
        "",
        "## Identical-coverage clusters (parametrize candidates)",
        "",
    ]
    if report["identical_clusters"]:
        out.append("| # | tests | lines each |")
        out.append("|---|---|---|")
        for i, c in enumerate(report["identical_clusters"], 1):
            out.append(f"| {i} | {', '.join(c['tests'])} | {c['lines_each']} |")
    else:
        out.append("_None — no two tests share an identical line set._")
    return "\n".join(out) + "\n"


def _heatmap(test_lines: dict, out_path: Path) -> str | None:
    """Optional raster Jaccard-similarity heatmap. Write-only for the agent, so
    it's behind --viz and degrades gracefully if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    names = sorted(test_lines)
    n = len(names)
    sets = [test_lines[k] for k in names]
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j]) or 1
            mat[i][j] = inter / union
    fig, ax = plt.subplots(figsize=(max(6, n * 0.18), max(6, n * 0.18)))
    ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(
        "Test-to-test line-coverage Jaccard similarity\n"
        "(shape only — confirm assertions before merging)",
        fontsize=8,
    )
    short_names = [t.rsplit(".", 1)[-1] for t in names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=90, fontsize=4)
    ax.set_yticklabels(short_names, fontsize=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        required=True,
        help="Dotted module(s) to scope coverage to, e.g. "
        "my_pkg.my_feature (repeatable / comma-sep).",
    )
    ap.add_argument(
        "--tests",
        action="append",
        default=[],
        required=True,
        help="pytest path(s) to run (repeatable / comma-sep).",
    )
    ap.add_argument(
        "--out", required=True, help="Output directory (created if missing)."
    )
    ap.add_argument(
        "--preimport",
        action="append",
        default=[],
        help="Native module(s) to import UNINSTRUMENTED before arming "
        "coverage — use when importing the feature aborts the process "
        "under the tracer (a pybind11/Cython extension). Default: none.",
    )
    ap.add_argument(
        "--branch",
        action="store_true",
        help="Enable branch coverage (sharper, slightly noisier).",
    )
    ap.add_argument(
        "--viz",
        action="store_true",
        help="Also emit a raster heatmap (PNG) of test-to-test "
        "similarity. Needs matplotlib; skipped with a note if "
        "absent. Text artifacts (Mermaid + markdown) are always "
        "written regardless of this flag.",
    )
    args = ap.parse_args()

    sources = _split(args.source)
    tests = _split(args.tests)
    preimports = _split(args.preimport)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load any --preimport native modules AND the feature's full import tree
    #    UNINSTRUMENTED. Importing the feature pulls in every transitive native
    #    module the tests will touch, so none of them load later under the armed
    #    tracer -- which is what aborts the process. `--preimport` is a belt-and-
    #    braces fallback for native modules a test imports directly but the feature
    #    doesn't (omit it entirely for pure-Python projects).
    import importlib

    for mod in [*preimports, *sources]:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass

    # 2. Arm coverage, scoped to the feature, with per-test dynamic contexts.
    import coverage

    cov = coverage.Coverage(
        source=sources, branch=args.branch, data_file=str(out / ".coverage")
    )
    cov.set_option("run:dynamic_context", "test_function")
    cov.start()

    # 3. Evict ONLY the pure-Python source modules so pytest re-imports them under
    #    the tracer (their import-time lines get measured -- no "module-not-measured"
    #    warning). Their native/3rd-party deps stay cached in sys.modules, so nothing
    #    pybind11 re-loads under coverage and the process does not abort.
    for name in list(sys.modules):
        if any(name == s or name.startswith(s + ".") for s in sources):
            del sys.modules[name]

    # 4. Run the feature's tests in-process (NOT the --cov plugin: -p no:cov).
    import pytest

    rc = pytest.main([*tests, "-q", "-p", "no:cacheprovider", "-p", "no:cov"])
    cov.stop()
    cov.save()

    # 5. Build the line<->test maps from the recorded contexts.
    data = cov.get_data()
    # test context -> set of (file, lineno) it executed
    test_lines: dict[str, set[tuple[str, int]]] = defaultdict(set)
    # (file, lineno) -> set of test contexts that executed it
    line_tests: dict[tuple[str, int], set[str]] = defaultdict(set)
    for f in data.measured_files():
        for lineno, contexts in data.contexts_by_lineno(f).items():
            for ctx in contexts:
                if not ctx:  # "" == import-time / no test running
                    continue
                test_lines[ctx].add((f, lineno))
                line_tests[(f, lineno)].add(ctx)

    # Identical-signature clusters: tests covering the exact same line set.
    by_signature: dict[frozenset[tuple[str, int]], list[str]] = defaultdict(list)
    for ctx, lines in test_lines.items():
        by_signature[frozenset(lines)].append(ctx)
    identical_clusters = sorted(
        ([sorted(g), len(sig)] for sig, g in by_signature.items() if len(g) > 1),
        key=lambda x: -len(x[0]),
    )

    # Subset pairs: test A's lines are a strict subset of test B's (A redundant
    # w.r.t. coverage — still confirm assertions before merging).
    subset_pairs: list[dict[str, object]] = []
    items = list(test_lines.items())
    for a, la in items:
        for b, lb in items:
            if a != b and la < lb:
                subset_pairs.append({"subset": a, "superset": b, "lines": len(la)})

    # Sole-owner lines: a line executed by exactly one test => that test is
    # load-bearing for it. Count per test.
    sole_owner_count: dict[str, int] = defaultdict(int)
    for (_f, _ln), owners in line_tests.items():
        if len(owners) == 1:
            sole_owner_count[next(iter(owners))] += 1
    # Tests with zero sole-owner lines are pure-redundant candidates.
    pure_redundant = sorted(t for t in test_lines if sole_owner_count.get(t, 0) == 0)

    # Per-file numeric summary + missing lines (via coverage's own analysis).
    files_summary = []
    total_stmts = total_miss = 0
    for f in sorted(data.measured_files()):
        _, statements, _excluded, missing, _ = cov.analysis2(f)
        total_stmts += len(statements)
        total_miss += len(missing)
        files_summary.append(
            {
                "file": f,
                "statements": len(statements),
                "missing": len(missing),
                "missing_lines": sorted(missing),
                "percent": round(
                    100 * (len(statements) - len(missing)) / max(1, len(statements)), 1
                ),
            }
        )
    total_pct = round(100 * (total_stmts - total_miss) / max(1, total_stmts), 1)

    def short(t: str) -> str:
        return t.rsplit(".", 1)[-1]

    report = {
        "pytest_returncode": int(rc),
        "branch": args.branch,
        "sources": sources,
        "tests": tests,
        "total_percent": total_pct,
        "total_statements": total_stmts,
        "total_missing": total_miss,
        "files": files_summary,
        "num_tests": len(test_lines),
        "identical_clusters": [
            {"tests": [short(t) for t in g], "lines_each": n}
            for g, n in identical_clusters
        ],
        "subset_pairs": [
            {
                "subset": short(p["subset"]),
                "superset": short(p["superset"]),
                "lines": p["lines"],
            }
            for p in subset_pairs
        ],
        "pure_redundant_tests": [short(t) for t in pure_redundant],
        # False when no test sole-owns a line (module too small for line coverage
        # to distinguish tests) -> trust the clusters + assertions, not this list.
        "line_coverage_discriminates": len(pure_redundant) != len(test_lines),
        "sole_owner_lines_per_test": {
            short(t): n for t, n in sorted(sole_owner_count.items())
        },
    }
    (out / "coverage.json").write_text(json.dumps(report, indent=2))

    # 6. HTML with per-line contexts (hover a line -> the tests that hit it).
    try:
        cov.html_report(directory=str(out / "htmlcov"), show_contexts=True)
    except Exception as exc:  # noqa: BLE001 - html is a nicety, never fatal
        report["html_error"] = str(exc)

    # 7. Visualizations of the test-to-test cluster structure. Text formats
    #    (Mermaid + markdown) are always emitted — the LLM reviewer reads/embeds
    #    them; raster (PNG) is write-only for the agent so it is behind --viz.
    #    Every artifact carries the discriminates guardrail in its caption.
    caption = _caption(report["line_coverage_discriminates"])
    clustered = {t for c in report["identical_clusters"] for t in c["tests"]}
    singletons = sorted(short(t) for t in test_lines if short(t) not in clustered)
    clusters_short = [
        (c["tests"], c["lines_each"]) for c in report["identical_clusters"]
    ]
    mermaid = _mermaid(clusters_short, report["subset_pairs"], singletons, caption)
    (out / "clusters.mmd").write_text(mermaid + "\n")
    (out / "coverage.md").write_text(_markdown(report, mermaid, caption))
    if args.viz:
        png = _heatmap(test_lines, out / "similarity_heatmap.png")
        report["heatmap"] = png or "skipped (matplotlib not installed)"

    # 8. Concise human summary.
    print(f"\n==== COVERAGE CONTEXTS ({', '.join(sources)}) ====")
    print(
        f"pytest rc={rc}  |  {len(test_lines)} tests  |  "
        f"line coverage {total_pct}%  ({total_miss}/{total_stmts} missing)"
    )
    for fs in files_summary:
        miss = fs["missing_lines"]
        # Normalise separators so the display is the same on POSIX and Windows.
        disp = fs["file"].replace("\\", "/")
        disp = disp.split("site-packages/", 1)[-1]
        print(f"  {disp}: {fs['percent']}%" + (f"  missing {miss}" if miss else ""))
    if identical_clusters:
        print(
            "\nIdentical-coverage clusters — same lines, so candidates to collapse "
            "into ONE @pytest.mark.parametrize (only if their assertions differ "
            "merely in data, not in behavior):"
        )
        for g, n in identical_clusters:
            print(f"  - {{{', '.join(short(t) for t in g)}}}  ({n} lines each)")
    # Sole-owner lines mark load-bearing tests; when NO test owns a line (common
    # for a small module where one body serves many behaviors) line coverage
    # cannot discriminate tests — say so rather than flagging everything.
    if pure_redundant and len(pure_redundant) == len(test_lines):
        print(
            f"\nNote: none of the {len(test_lines)} tests sole-owns a line "
            "(module is small / many behaviors share the same lines). Line "
            "coverage cannot prove redundancy here — judge merges by the "
            "clusters above PLUS the assertions, not by coverage alone."
        )
    elif pure_redundant:
        print(
            "\nPure-redundant tests (cover no line that another test doesn't — "
            "confirm assertions are subsumed, then merge):"
        )
        for t in pure_redundant:
            print(f"  - {short(t)}")
    print(
        f"\nWrote {out / 'coverage.json'}, {out / 'coverage.md'} (with Mermaid), "
        f"{out / 'clusters.mmd'}, and {out / 'htmlcov'}/index.html"
        + (f", {report.get('heatmap')}" if args.viz else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
