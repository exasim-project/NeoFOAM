# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Render the study's per-case results into one self-contained ``report.html``.

No template engine and no plotting library — neither is a repo dependency — just
string assembly with everything (CSS + a tiny sort script) inlined, so the file
opens anywhere with no assets alongside it.

The headline is three counts over three row groups — **matched** (reproduces
native within tolerance, strictly or to round-off), **differs** (ran, but a field
or the post-solve cell count disagrees), **failed** (the neofoam solver did not
finish). Everything else is not a result: a harness fault (:data:`HARNESS_FAULTS`)
or a case that should never have been selected (``UNSUPPORTED_CASE`` — the
config's ``exclude:`` channel exists for it). Both render outside the row groups.

The per-row ``max abs``/``max rel`` columns and the outcome legend are diagnostic
scaffolding, rendered only with ``diagnostic=True`` (the report subcommand's
``--diagnostic`` flag).
"""

from __future__ import annotations

import html
from collections import Counter
from typing import Any
from urllib.parse import quote

from verification.dropin.cases import Study
from verification.dropin.execute import (
    CASE_SETUP_FAILED,
    COMPARE_FAILED,
    FIELDS_DIFFER,
    MATCHED,
    MATCHED_TO_ROUNDOFF,
    MESH_DIFFERS,
    MESH_NOT_REPRODUCIBLE,
    NATIVE_FAILED,
    SOLVER_FAILED,
    UNSUPPORTED_CASE,
)

__all__ = ["HARNESS_FAULTS", "harness_faults", "render_report"]

#: Outcome → the three-state fold. ``MESH_DIFFERS`` is a post-solve cell-count
#: divergence — a real difference, not a fault (the pre-solve mismatch is
#: ``MESH_NOT_REPRODUCIBLE``). ``TIMEOUT`` survives only in archived records; a
#: killed run did not finish, so it folds with failed.
_STATE_OF = {
    MATCHED: "matched",
    MATCHED_TO_ROUNDOFF: "matched",
    FIELDS_DIFFER: "differs",
    MESH_DIFFERS: "differs",
    SOLVER_FAILED: "failed",
    "TIMEOUT": "failed",
}

#: Broken runs, not results: the harness (not the solver under test) is at fault,
#: so these never appear as table rows and must fail the report step.
HARNESS_FAULTS = frozenset(
    {NATIVE_FAILED, CASE_SETUP_FAILED, COMPARE_FAILED, MESH_NOT_REPRODUCIBLE}
)

_STATE_TITLES = (
    ("matched", "Matched"),
    ("differs", "Differs"),
    ("failed", "Failed"),
)

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
    "MESH_DIFFERS": "warn",
    "MESH_NOT_REPRODUCIBLE": "muted",
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
        "The case can't be run as a drop-in — no unique solver token in Allrun to "
        "swap, Allrun needs a solver mode neofoam does not implement, or the case "
        "needs a capability neofoam refuses (AMR) — so it is outside this study's scope."
    ),
    "CASE_SETUP_FAILED": "casebuild staging aborted before the run — a harness fault.",
    "COMPARE_FAILED": (
        "The run finished but the post-run field read/compare failed — a harness fault."
    ),
    "MESH_DIFFERS": (
        "The solve ended with different cell counts on the two sides (e.g. AMR), "
        "so the final fields aren't directly comparable — a real solver difference."
    ),
    "MESH_NOT_REPRODUCIBLE": (
        "The meshes already differed before the solve — a nondeterministic mesher, "
        "a harness fault, not a solver verdict."
    ),
    # No longer producible: the run rule is plain shell with no timeout (a batch
    # scheduler caps wall time instead). Kept so a report regenerated from archived
    # results that predate that change still renders its legend.
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


def _simplify_details(row: dict[str, Any]) -> str:
    """The "simplified" badge plus a collapsible reason + patched entries, or ``""``.

    A simplified case ran with settings the study substituted on *both* sides, so its
    verdict is about the simplified case — the badge says so on the row itself and the
    expander says exactly what was changed.
    """
    simplify = row.get("simplify") or {}
    if not simplify:
        return ""
    patched = "\n".join(
        "\n".join([rel] + [f"    {key}: {value}" for key, value in overrides.items()])
        for rel, overrides in simplify.get("patch", {}).items()
    )
    return (
        ' <span class="pill muted">simplified</span>'
        '<details class="log"><summary>simplified on both sides</summary>'
        f'<p class="muted">{_esc(simplify.get("reason", ""))}</p>'
        f'<pre class="log">{_esc(patched)}</pre></details>'
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
        "simplify": record.get("simplify") or {},
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


def _fold(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into the three states, the harness faults, and the unsupported.

    Every row lands in exactly one bucket; an outcome this vocabulary does not
    know raises rather than silently dropping a case from the arithmetic.
    """
    states: dict[str, list[dict[str, Any]]] = {"matched": [], "differs": [], "failed": []}
    faults: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for row in rows:
        outcome = row["outcome"]
        if outcome in HARNESS_FAULTS:
            faults.append(row)
        elif outcome == UNSUPPORTED_CASE:
            unsupported.append(row)
        elif outcome in _STATE_OF:
            states[_STATE_OF[outcome]].append(row)
        else:
            raise ValueError(f"unknown outcome {outcome!r} for case {row['name']!r}")
    return states, faults, unsupported


def _rows(rows: list[dict[str, Any]], diagnostic: bool) -> str:
    ordered = sorted(rows, key=lambda r: (r["name"], r.get("label", "")))
    out = []
    for row in ordered:
        cells = (
            "<tr>"
            f'<td data-sort="{_esc(row["name"])}"><code>{_esc(row["name"])}</code>'
            f"{_simplify_details(row)}</td>"
            f"<td><code>{_esc(row.get('label', ''))}</code></td>" + _outcome_cell(row)
        )
        if diagnostic:
            cells += _num_cell(float(row.get("worst_abs", 0.0)))
            cells += _num_cell(float(row.get("worst_rel", 0.0)))
        out.append(cells + "</tr>")
    return "\n".join(out)


def _table(rows: list[dict[str, Any]], diagnostic: bool) -> str:
    header = "<tr><th>Case</th><th>Backend</th><th>Outcome</th>"
    if diagnostic:
        header += "<th>max abs</th><th>max rel</th>"
    header += "</tr>"
    return (
        '<table class="sortable">\n<thead>'
        + header
        + "</thead>\n<tbody>\n"
        + _rows(rows, diagnostic)
        + "\n</tbody>\n</table>"
    )


def _headline(states: dict[str, list[dict[str, Any]]], total: int) -> str:
    counts = " &nbsp; ".join(
        f"{title}: <b>{len(states[state])}</b>" for state, title in _STATE_TITLES
    )
    return f"<p>Case/backend runs: <b>{total}</b> &nbsp; {counts}</p>"


def _legend(rows: list[dict[str, Any]]) -> str:
    """The per-outcome count table with one-line meanings — diagnostic only."""
    counts = Counter(row["outcome"] for row in rows)
    rows_html = "\n".join(
        f'<tr><td><span class="pill {_OUTCOME_CLASS.get(outcome, "muted")}">'
        f'{_esc(outcome)}</span></td><td class="num">{count}</td>'
        f'<td class="muted">{_esc(_OUTCOME_HELP.get(outcome, ""))}</td></tr>'
        for outcome, count in counts.most_common()
    )
    return (
        '<table class="summary"><thead><tr>'
        "<th>Outcome</th><th>Count</th><th>Meaning</th>"
        "</tr></thead><tbody>\n" + rows_html + "\n</tbody></table>"
    )


def _not_result_item(row: dict[str, Any]) -> str:
    pill = f'<span class="pill {_OUTCOME_CLASS.get(row["outcome"], "muted")}">'
    pill += f"{_esc(row['outcome'])}</span>"
    detail = f' <span class="muted">{_esc(row["detail"])}</span>' if row.get("detail") else ""
    return (
        f"<li><code>{_esc(row['name'])}</code> <code>{_esc(row.get('label', ''))}</code> "
        f"{pill}{detail}{_simplify_details(row)}{_log_details(row)}</li>"
    )


def _fault_section(faults: list[dict[str, Any]]) -> str:
    """Harness faults are broken runs, not results — a list apart, never table rows."""
    items = "\n".join(_not_result_item(row) for row in faults)
    return (
        f"<h2>Harness faults ({len(faults)})</h2>\n"
        '<p class="muted">Broken runs, not results: the harness — not the solver '
        "under test — is at fault. These must be fixed, not scored.</p>\n"
        f"<ul>\n{items}\n</ul>"
    )


def _unsupported_section(rows: list[dict[str, Any]]) -> str:
    items = "\n".join(_not_result_item(row) for row in rows)
    return (
        f"<h2>Unsupported cases ({len(rows)})</h2>\n"
        '<p class="muted">These cases cannot run as drop-ins and should be moved '
        "to the config's <code>exclude:</code> list with their reason.</p>\n"
        f"<ul>\n{items}\n</ul>"
    )


def harness_faults(study: Study, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The fault rows in *records* — broken runs the report refuses to score.

    Same row set the report renders (stale records for undiscovered cases are
    dropped, legacy outcome strings aliased), so the report step can fail on
    exactly the faults its own fault section shows.
    """
    known = {case.id for case in study.cases}
    rows = [row for record in records if record["id"] in known for row in _rows_of(record)]
    return [row for row in rows if row["outcome"] in HARNESS_FAULTS]


def render_report(study: Study, records: list[dict[str, Any]], diagnostic: bool = False) -> str:
    """Assemble the full HTML document for *study* from its per-case *records*.

    ``diagnostic=True`` adds the per-row ``max abs``/``max rel`` columns and the
    per-outcome legend; the default report is the bare three-state fold.
    """
    by_id = {case.id: case for case in study.cases}
    # Drop results for cases no longer discovered (e.g. a retired native solver), so
    # the report always reflects the current study rather than stale on-disk runs.
    records = [record for record in records if record["id"] in by_id]
    all_rows = [row for record in records for row in _rows_of(record)]
    states, faults, unsupported = _fold(all_rows)

    sections = [f"<h2>Summary</h2>\n{_headline(states, len(all_rows))}"]
    if diagnostic:
        sections.append(_legend(all_rows))
    sections.extend(
        f"<h2>{title} ({len(states[state])})</h2>\n{_table(states[state], diagnostic)}"
        for state, title in _STATE_TITLES
        if states[state]
    )
    if faults:
        sections.append(_fault_section(faults))
    if unsupported:
        sections.append(_unsupported_section(unsupported))

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
