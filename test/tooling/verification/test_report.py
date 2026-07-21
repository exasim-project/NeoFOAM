# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The HTML report renders every discovered case, matched or not.

Pure string assembly — no OpenFOAM, no solver run — so it guards the report shape
on every ``pytest`` run. The heavy sweep is Snakemake-only and lives outside
``testpaths``. The one invariant worth pinning: a ``FIELDS_DIFFER`` row must show
both ``max abs`` and ``max rel``, because at ``atol=1e-15`` round-off reads as a
difference and only the relative figure tells equivalence from a real defect.
"""

from pathlib import Path

from neofoam.tooling.verification.report import render_report
from neofoam.tooling.verification.study import Case, Study


def _study(cases: list[Case]) -> Study:
    return Study(
        title="demo study",
        config_path=Path("config.yaml"),
        cases=cases,
        tier_titles={"A": "Tier A — runnable", "D": "Tier D — blocked"},
        config={},
    )


def _case(name: str, tier: str) -> Case:
    return Case(
        id=name.replace("/", "__"),
        name=name,
        path=Path("/tutorials") / name,
        native_solver="simpleFoam",
        app="neofoam solver incompressiblefluid",
        fields=("U", "p"),
        turbulence="kEpsilon",
        tier=tier,
        reason="" if tier == "A" else "dynamic mesh",
    )


def _candidate(label: str, outcome: str, **over: object) -> dict[str, object]:
    return {
        "label": label,
        "app": f"neofoam solver {label}",
        "outcome": outcome,
        "detail": over.get("detail", ""),
        "worst_abs": over.get("worst_abs", 0.0),
        "worst_rel": over.get("worst_rel", 0.0),
        "fields": over.get("fields", []),
        "log_path": over.get("log_path", ""),
        "log_tail": over.get("log_tail", ""),
    }


def test_report_lists_matched_and_diverged_with_both_tolerances() -> None:
    cases = [_case("simpleFoam/pitzDaily", "A"), _case("simpleFoam/mixer", "D")]
    records = [
        {
            "id": "simpleFoam__pitzDaily",
            "name": "simpleFoam/pitzDaily",
            "native_solver": "simpleFoam",
            "tier": "A",
            "predicted_blocker": "",
            "turbulence": "kEpsilon",
            "parallel": False,
            "candidates": [
                _candidate(
                    "incompressiblefluid",
                    "MATCHED",
                    fields=[{"name": "U", "matched": True, "abs": 0.0, "rel": 0.0}],
                )
            ],
        },
        {
            "id": "simpleFoam__mixer",
            "name": "simpleFoam/mixer",
            "native_solver": "simpleFoam",
            "tier": "D",
            "predicted_blocker": "dynamic mesh",
            "turbulence": "kEpsilon",
            "parallel": True,
            "candidates": [
                _candidate(
                    "incompressiblefluid",
                    "FIELDS_DIFFER",
                    detail="U",
                    worst_abs=1.2e-11,
                    worst_rel=1.8e-13,
                    fields=[
                        {"name": "U", "matched": False, "abs": 1.2e-11, "rel": 1.8e-13}
                    ],
                )
            ],
        },
    ]

    html = render_report(_study(cases), records)

    assert "<title>demo study</title>" in html
    assert "Matched the native reference: <b>1</b>" in html
    # One flat list — both cases render in a single table, not per-tier sections.
    assert "simpleFoam/pitzDaily" in html and "simpleFoam/mixer" in html
    assert html.count('<table class="sortable">') == 1
    assert "Tier A — runnable" not in html and "Tier D — blocked" not in html
    # The FIELDS_DIFFER row shows abs AND rel, so round-off is distinguishable.
    assert "1.20e-11" in html and "1.80e-13" in html


def test_report_shows_one_row_per_backend() -> None:
    """A case compared against two backends renders two rows, one per backend."""
    cases = [_case("simpleFoam/pitzDaily", "A")]
    records = [
        {
            "id": "simpleFoam__pitzDaily",
            "name": "simpleFoam/pitzDaily",
            "native_solver": "simpleFoam",
            "tier": "A",
            "predicted_blocker": "",
            "turbulence": "kEpsilon",
            "parallel": False,
            "candidates": [
                _candidate("incompressiblefluid", "MATCHED"),
                _candidate("incompressiblefluidneon", "FIELDS_DIFFER", worst_rel=1e-1),
            ],
        }
    ]

    html = render_report(_study(cases), records)

    # Both backends appear, and the summary counts two case/backend runs.
    assert "incompressiblefluid</code>" in html
    assert "incompressiblefluidneon</code>" in html
    assert "Case/backend runs: <b>2</b>" in html


def test_report_attaches_the_log_for_a_crashed_case() -> None:
    """A crashed case embeds its log tail inline and links the full log file."""
    cases = [_case("simpleFoam/broken", "D")]
    records = [
        {
            "id": "simpleFoam__broken",
            "name": "simpleFoam/broken",
            "native_solver": "simpleFoam",
            "tier": "D",
            "predicted_blocker": "fvOptions sources",
            "turbulence": "kEpsilon",
            "parallel": False,
            "candidates": [
                _candidate(
                    "incompressiblefluid",
                    "SOLVER_FAILED",
                    detail="NotImplementedError: no fvOptions",
                    log_path=(
                        "work/simpleFoam__broken/incompressiblefluid/"
                        "log.neofoam solver incompressiblefluid"
                    ),
                    log_tail=(
                        "Traceback (most recent call last):\nNotImplementedError: nope"
                    ),
                )
            ],
        }
    ]

    html = render_report(_study(cases), records)

    assert "<details" in html and "NotImplementedError: nope" in html
    # The full-log link is URL-encoded (the neofoam log basename has spaces).
    assert "log.neofoam%20solver%20incompressiblefluid" in html


def test_report_flags_cases_without_a_result() -> None:
    """A discovered case with no result file is named, not silently dropped."""
    cases = [_case("simpleFoam/pitzDaily", "A"), _case("simpleFoam/missing", "A")]
    records = [
        {
            "id": "simpleFoam__pitzDaily",
            "name": "simpleFoam/pitzDaily",
            "native_solver": "simpleFoam",
            "tier": "A",
            "predicted_blocker": "",
            "turbulence": "kEpsilon",
            "parallel": False,
            "candidates": [_candidate("incompressiblefluid", "MATCHED")],
        }
    ]

    html = render_report(_study(cases), records)

    assert "No result recorded for" in html
    assert "simpleFoam/missing" in html


def test_report_summary_explains_outcomes() -> None:
    """The Summary is a legend: each outcome's one-line meaning appears beside it."""
    cases = [_case("simpleFoam/pitzDaily", "A"), _case("simpleFoam/mixer", "D")]
    records = [
        {
            "id": "simpleFoam__pitzDaily",
            "name": "simpleFoam/pitzDaily",
            "native_solver": "simpleFoam",
            "tier": "A",
            "predicted_blocker": "",
            "turbulence": "kEpsilon",
            "parallel": False,
            "candidates": [_candidate("incompressiblefluid", "MATCHED")],
        },
        {
            "id": "simpleFoam__mixer",
            "name": "simpleFoam/mixer",
            "native_solver": "simpleFoam",
            "tier": "D",
            "predicted_blocker": "",
            "turbulence": "kEpsilon",
            "parallel": False,
            "candidates": [_candidate("incompressiblefluid", "FIELDS_DIFFER")],
        },
    ]

    html = render_report(_study(cases), records)

    assert "Meaning" in html
    assert "differs beyond tolerance" in html  # the FIELDS_DIFFER meaning
    assert "field-for-field within tolerance" in html  # the MATCHED meaning


def test_report_aliases_legacy_outcome_strings() -> None:
    """A cached result with an old outcome name renders under the current label."""
    cases = [_case("simpleFoam/mixer", "D")]
    records = [
        {
            "id": "simpleFoam__mixer",
            "name": "simpleFoam/mixer",
            "native_solver": "simpleFoam",
            "tier": "D",
            "predicted_blocker": "",
            "turbulence": "kEpsilon",
            "parallel": False,
            "candidates": [_candidate("incompressiblefluid", "DIVERGED")],
        }
    ]

    html = render_report(_study(cases), records)

    assert "FIELDS_DIFFER" in html
    assert "DIVERGED" not in html
