# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Render the study's per-case results into one self-contained ``report.html``.

No template engine and no plotting library — neither is a repo dependency — just
string assembly with everything (CSS + a tiny sort script) inlined, so the file
opens anywhere with no assets alongside it.

The ``max rel`` column carries real weight, not decoration. The pass criterion is
strict (``atol=1e-15``), so round-off in near-zero field entries reads as
``FIELDS_DIFFER``. A reader must be able to tell a 1e-13 disagreement (numerically
equivalent) from a 1e-1 one (a genuine defect) — both columns are always shown.
"""

from __future__ import annotations

import html
from collections import Counter
from typing import Any
from urllib.parse import quote

from neofoam.tooling.verification.study import Study

__all__ = ["render_report"]

#: Outcome → CSS class, so the table colour-codes at a glance.
_OUTCOME_CLASS = {
    "MATCHED": "ok",
    "MATCHED_TO_ROUNDOFF": "ok",
    "FIELDS_DIFFER": "warn",
    "SOLVER_FAILED": "bad",
    "NATIVE_FAILED": "muted",
    "UNSUPPORTED_CASE": "muted",
    "CASE_SETUP_FAILED": "muted",
    "COMPARE_FAILED": "muted",
    "MESH_DIFFERS": "muted",
    "TIMEOUT": "bad",
}

#: Outcome → one-line meaning, rendered beside the count so the Summary is a legend
#: a reader can act on without knowing the harness internals.
_OUTCOME_HELP = {
    "MATCHED": "Reproduced the native reference field-for-field within tolerance.",
    "MATCHED_TO_ROUNDOFF": (
        "Every differing field is within rel < 1e-10 — reproduces native to machine "
        "precision, just outside the strict atol=1e-15."
    ),
    "FIELDS_DIFFER": (
        "Ran to completion but a field differs beyond tolerance — read max rel: "
        "~1e-13 is round-off, ~1e0 is a real defect."
    ),
    "SOLVER_FAILED": "The neofoam solver crashed or errored; the detail + log say why.",
    "NATIVE_FAILED": (
        "The native reference itself did not run — a harness fault, not a solver verdict."
    ),
    "UNSUPPORTED_CASE": (
        "No unique solver token in Allrun to swap — the case can't be run as a "
        "drop-in, so it is outside this study's scope."
    ),
    "CASE_SETUP_FAILED": "casebuild staging aborted before the run — a harness fault.",
    "COMPARE_FAILED": (
        "The run finished but the post-run field read/compare failed — a harness fault."
    ),
    "MESH_DIFFERS": "Native and candidate meshes differ, so the fields aren't comparable.",
    "TIMEOUT": "The run exceeded its wall-clock limit and was killed.",
}

#: Legacy outcome strings in cached results/*.json → current name, so a report-only
#: regeneration renders the new labels without re-running any solver. Deletable once
#: a full re-sweep has rewritten every cached result with the new names.
_OUTCOME_ALIASES = {
    "DIVERGED": "FIELDS_DIFFER",
    "NO_SWAP_POINT": "UNSUPPORTED_CASE",
    "STAGE_FAILED": "CASE_SETUP_FAILED",
}

_CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.5 system-ui, sans-serif; margin: 2rem; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; }
th, td { padding: 4px 8px; border-bottom: 1px solid #8884; text-align: left; }
th { cursor: pointer; user-select: none; white-space: nowrap; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code { font-family: ui-monospace, monospace; }
.ok    { color: #1a7f37; } .warn { color: #9a6700; }
.bad   { color: #cf222e; } .muted { color: #57606a; }
.pill  { display: inline-block; padding: 0 6px; border-radius: 6px; font-size: 12px;
         border: 1px solid currentColor; }
.summary td.num { font-weight: 600; }
details.log { margin-top: 4px; }
details.log summary { color: #57606a; cursor: pointer; }
pre.log { max-height: 22rem; overflow: auto; margin: 4px 0 0; padding: 8px;
          background: #8881; border-radius: 6px; font-size: 12px; line-height: 1.3;
          white-space: pre; }
"""

# Small dependency-free column sort: click a header, rows reorder by that column.
_SCRIPT = """
document.querySelectorAll('table.sortable').forEach(function (table) {
  table.querySelectorAll('th').forEach(function (th, col) {
    th.addEventListener('click', function () {
      var tbody = table.tBodies[0];
      var rows = Array.prototype.slice.call(tbody.rows);
      var asc = !(th.dataset.asc === 'true');
      th.dataset.asc = asc;
      rows.sort(function (a, b) {
        var x = a.cells[col].dataset.sort ?? a.cells[col].innerText;
        var y = b.cells[col].dataset.sort ?? b.cells[col].innerText;
        var nx = parseFloat(x), ny = parseFloat(y);
        if (!isNaN(nx) && !isNaN(ny)) { return asc ? nx - ny : ny - nx; }
        return asc ? String(x).localeCompare(y) : String(y).localeCompare(x);
      });
      rows.forEach(function (r) { tbody.appendChild(r); });
    });
  });
});
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _log_details(record: dict[str, Any]) -> str:
    """A collapsible log for a crashed case: the tail inline, plus a full-log link.

    Inline so the report stays self-contained even if the run dir is later
    pruned; the link is a convenience for when it is still on disk.
    """
    tail = record.get("log_tail") or ""
    if not tail:
        return ""
    path = record.get("log_path") or ""
    link = f'<a href="{quote(path)}">full log</a>' if path else ""
    return (
        '<details class="log"><summary>log</summary>'
        f'{link}<pre class="log">{_esc(tail)}</pre></details>'
    )


def _rows_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    """One display row per candidate backend, carrying the case-level context.

    A result file holds every backend's diff against the shared native reference;
    the table shows one row per (case, backend), so a case with two backends is
    two rows differing only in the Backend column and the outcome/tolerances.
    """
    shared = {
        "name": record["name"],
        "native_solver": record["native_solver"],
        "turbulence": record["turbulence"],
        "parallel": record.get("parallel", False),
        "predicted_blocker": record.get("predicted_blocker", ""),
    }
    rows = [{**shared, **candidate} for candidate in record.get("candidates", [])]
    for row in rows:
        row["outcome"] = _OUTCOME_ALIASES.get(row["outcome"], row["outcome"])
    return rows


def _outcome_cell(row: dict[str, Any]) -> str:
    outcome = row["outcome"]
    css = _OUTCOME_CLASS.get(outcome, "muted")
    label = f'<span class="pill {css}">{_esc(outcome)}</span>'
    detail = row.get("detail", "")
    if detail:
        label += f' <span class="muted">{_esc(detail)}</span>'
    return f'<td data-sort="{_esc(outcome)}">{label}{_log_details(row)}</td>'


def _num_cell(value: float) -> str:
    return f'<td class="num" data-sort="{value:.6e}">{value:.2e}</td>'


def _rows(rows: list[dict[str, Any]]) -> str:
    # Passing rows first, then by name — the eye should land on what reproduced
    # native (strictly, or to round-off).
    passed = {"MATCHED", "MATCHED_TO_ROUNDOFF"}
    ordered = sorted(
        rows,
        key=lambda r: (r["outcome"] not in passed, r["name"], r.get("label", "")),
    )
    out = []
    for row in ordered:
        out.append(
            "<tr>"
            f"<td><code>{_esc(row['name'])}</code></td>"
            f"<td><code>{_esc(row.get('label', ''))}</code></td>"
            + _outcome_cell(row)
            + _num_cell(float(row.get("worst_abs", 0.0)))
            + _num_cell(float(row.get("worst_rel", 0.0)))
            + "</tr>"
        )
    return "\n".join(out)


def _table(rows: list[dict[str, Any]]) -> str:
    header = (
        "<tr><th>Case</th><th>Backend</th><th>Outcome</th><th>max abs</th><th>max rel</th></tr>"
    )
    return (
        '<table class="sortable">\n<thead>'
        + header
        + "</thead>\n<tbody>\n"
        + _rows(rows)
        + "\n</tbody>\n</table>"
    )


def _summary(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["outcome"] for row in rows)
    matched = counts.get("MATCHED", 0)
    roundoff = counts.get("MATCHED_TO_ROUNDOFF", 0)
    rows_html = "\n".join(
        f'<tr><td><span class="pill {_OUTCOME_CLASS.get(outcome, "muted")}">'
        f'{_esc(outcome)}</span></td><td class="num">{count}</td>'
        f'<td class="muted">{_esc(_OUTCOME_HELP.get(outcome, ""))}</td></tr>'
        for outcome, count in counts.most_common()
    )
    # Strict MATCHED stays its own figure; the round-off matches are shown beside it
    # (not folded in) so machine-precision cases read as passes without loosening what
    # "Matched" means.
    roundoff_note = (
        f" &nbsp; Matched to round-off (rel &lt; 1e-10): <b>{roundoff}</b>" if roundoff else ""
    )
    return (
        f"<p>Case/backend runs: <b>{len(rows)}</b> &nbsp; "
        f"Matched the native reference: <b>{matched}</b>{roundoff_note}</p>"
        '<table class="summary"><thead><tr>'
        "<th>Outcome</th><th>Count</th><th>Meaning</th>"
        "</tr></thead><tbody>\n" + rows_html + "\n</tbody></table>"
    )


def render_report(study: Study, records: list[dict[str, Any]]) -> str:
    """Assemble the full HTML document for *study* from its per-case *records*."""
    by_id = {case.id: case for case in study.cases}
    # Drop results for cases no longer discovered (e.g. a retired native solver), so
    # the report always reflects the current study rather than stale on-disk runs.
    records = [record for record in records if record["id"] in by_id]
    all_rows = [row for record in records for row in _rows_of(record)]

    sections = [
        f"<h2>Summary</h2>\n{_summary(all_rows)}",
        f"<h2>All cases</h2>\n{_table(all_rows)}",
    ]

    # Note any case that was discovered but produced no result file.
    missing = sorted(set(by_id) - {record["id"] for record in records})
    if missing:
        names = ", ".join(_esc(by_id[m].name) for m in missing)
        sections.append(f'<p class="muted">No result recorded for: {names}</p>')

    body = "\n".join(sections)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(study.title)}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<h1>{_esc(study.title)}</h1>\n"
        f"{body}\n"
        f"<script>{_SCRIPT}</script>\n"
        "</body>\n</html>\n"
    )
