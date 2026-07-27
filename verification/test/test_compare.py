# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""How a field diff becomes an outcome, and how a decomposed run is diffed.

Pure logic — no OpenFOAM field read is exercised here (that is the sweep's job and
the integration proof's). The classifier (:func:`compare._classify`) and the
decomposed detector/aggregator (:func:`compare._is_decomposed`,
:func:`compare._compare_decomposed`) are the parts that decide *which* outcome a run
earns, so they are pinned directly against synthetic inputs the way the rest of the
verification tooling tests pin ``_decide`` and ``_status_from_rundir``.

Three outcomes are the point:

* ``MATCHED_TO_ROUNDOFF`` — a near-zero field entry differs above ``atol=1e-15`` while
  the peak-normalised ``rel`` stays below ``1e-10`` (the real damBreak ``p_rgh``
  situation): a machine-precision reproduction, not a defect.
* ``MESH_DIFFERS`` — two internal fields of different length (AMR changed the cell
  count): a mesh difference, reported as such instead of an ``inf`` FIELDS_DIFFER.
* the decomposed compare — wave tutorials leave only ``processor*/<t>/`` with no
  ``reconstructPar``, so ranks are diffed pairwise and the worst rank wins per field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.dropin import compare
from verification.dropin.compare import (
    ATOL,
    FieldDiff,
    _classify,
    _compare_decomposed,
    _is_decomposed,
    compare_field,
)
from verification.dropin.execute import (
    COMPARE_FAILED,
    FIELDS_DIFFER,
    MATCHED,
    MATCHED_TO_ROUNDOFF,
    MESH_DIFFERS,
)


def test_classify_matches_when_every_field_is_within_tolerance() -> None:
    ref = np.array([1.0, 2.0, 3.0])
    diff = compare_field(ref, ref.copy(), "U")

    outcome, detail = _classify([diff])

    assert outcome == MATCHED
    assert detail == ""


def test_classify_roundoff_when_a_near_zero_entry_differs_above_atol() -> None:
    """A 5e-13 wobble in a near-zero entry of a peak-1e4 field: rel ~ 5e-17.

    It fails the strict elementwise allclose (atol=1e-15 at the zero entry) yet the
    peak-normalised rel is far below 1e-10, so it is a round-off match, not a defect.
    """
    ref = np.array([1.0e4, 0.0, -2.0e3])
    candidate = np.array([1.0e4, 5.0e-13, -2.0e3])
    diff = compare_field(ref, candidate, "p_rgh")
    assert diff.matched is False and diff.rel_diff < 1e-10  # the precondition

    outcome, detail = _classify([diff])

    assert outcome == MATCHED_TO_ROUNDOFF
    assert detail == "p_rgh"


def test_classify_fields_differ_on_a_real_defect() -> None:
    ref = np.array([1.0, 2.0])
    candidate = np.array([1.0, 2.2])  # rel = 0.1, well above 1e-10
    diff = compare_field(ref, candidate, "U")

    outcome, _ = _classify([diff])

    assert outcome == FIELDS_DIFFER


def test_classify_mesh_differs_on_a_length_mismatch() -> None:
    """Different internal-field lengths (AMR) → MESH_DIFFERS, not inf FIELDS_DIFFER."""
    ref = np.array([1.0, 2.0, 3.0])
    candidate = np.array([1.0, 2.0])
    diff = compare_field(ref, candidate, "alpha.water")
    assert diff.rel_diff == float("inf")  # the short-circuit compare_field returns

    outcome, detail = _classify([diff])

    assert outcome == MESH_DIFFERS
    assert "alpha.water" in detail


def _touch_time(run: Path, sub: str, time: str, fields: tuple[str, ...]) -> None:
    """Create ``run/<sub>/<time>/<field>`` files so the dir looks like a written run."""
    time_dir = run / sub / time if sub else run / time
    time_dir.mkdir(parents=True, exist_ok=True)
    for name in fields:
        (time_dir / name).write_text("")


def test_is_decomposed_true_when_processors_carry_a_later_time(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _touch_time(run, "", "0", ("U",))  # root has only the initial conditions
    _touch_time(run, "processor0", "0", ("U",))
    _touch_time(run, "processor0", "2", ("U",))

    assert _is_decomposed(run) is True


def test_is_decomposed_false_for_a_serial_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _touch_time(run, "", "0", ("U",))
    _touch_time(run, "", "2", ("U",))

    assert _is_decomposed(run) is False


def test_is_decomposed_false_when_reconstructed(tmp_path: Path) -> None:
    """A reconstructed run has every time at the root, so it takes the serial path."""
    run = tmp_path / "run"
    _touch_time(run, "", "0", ("U",))
    _touch_time(run, "", "2", ("U",))
    _touch_time(run, "processor0", "0", ("U",))
    _touch_time(run, "processor0", "2", ("U",))

    assert _is_decomposed(run) is False


def _decomposed_pair(tmp_path: Path, ranks: int, fields: tuple[str, ...]) -> tuple[Path, Path]:
    native = tmp_path / "native"
    neo = tmp_path / "neo"
    for run in (native, neo):
        for r in range(ranks):
            _touch_time(run, f"processor{r}", "0", fields)
            _touch_time(run, f"processor{r}", "2", fields)
    return native, neo


def test_compare_decomposed_takes_the_worst_rank_per_field(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Two ranks: rank 0 matches, rank 1 has a defect — the field's diff is rank 1's."""
    native, neo = _decomposed_pair(tmp_path, ranks=2, fields=("U",))

    def fake_read(proc_dir: Path, system_src: Path, name: str, time: str) -> np.ndarray:
        base = np.array([1.0, 2.0])
        if system_src.name == "native":
            return base
        # neo side: rank 0 identical, rank 1 diverged
        return base if proc_dir.name == "processor0" else base + np.array([0.0, 0.3])

    monkeypatch.setattr(compare, "_read_decomposed_field", fake_read)

    outcome, detail, diffs = _compare_decomposed(native, neo, ["U"])

    assert outcome == FIELDS_DIFFER
    assert detail == "U"
    # One aggregated diff for U, carrying the worst (rank 1) disagreement.
    assert len(diffs) == 1
    np.testing.assert_allclose(
        diffs[0].rel_diff, 0.15, rtol=1e-12, err_msg="worst rank rel = 0.3 / peak 2.0"
    )


def test_compare_decomposed_matches_when_all_ranks_match(tmp_path: Path, monkeypatch: Any) -> None:
    native, neo = _decomposed_pair(tmp_path, ranks=2, fields=("U",))

    def fake_read(proc_dir: Path, system_src: Path, name: str, time: str) -> np.ndarray:
        return np.array([1.0, 2.0])

    monkeypatch.setattr(compare, "_read_decomposed_field", fake_read)

    outcome, _, diffs = _compare_decomposed(native, neo, ["U"])

    assert outcome == MATCHED
    assert all(d.matched for d in diffs)


def test_compare_decomposed_fails_on_processor_count_mismatch(tmp_path: Path) -> None:
    native = tmp_path / "native"
    neo = tmp_path / "neo"
    _touch_time(native, "processor0", "2", ("U",))
    _touch_time(native, "processor1", "2", ("U",))
    _touch_time(neo, "processor0", "2", ("U",))

    outcome, detail, diffs = _compare_decomposed(native, neo, ["U"])

    assert outcome == COMPARE_FAILED
    assert "processor count differs" in detail
    assert diffs == []


def test_field_diff_carries_the_diff_shape() -> None:
    """FieldDiff serialises the fields the report reads back."""
    diff = FieldDiff("U", matched=False, abs_diff=1.2e-11, rel_diff=1.8e-13)

    assert diff.as_dict() == {
        "name": "U",
        "matched": False,
        "abs": 1.2e-11,
        "rel": 1.8e-13,
    }


def test_a_numerically_zero_reference_field_reports_no_relative_error() -> None:
    """A field that is zero everywhere has no scale to normalise by.

    Regression: ``rel = abs / peak`` divided one round-off by another and reported
    ``rel = 1.0`` for a ``p_rgh`` matching to 1e-17 — indistinguishable in the
    report from total divergence, and the reason two archived MATCHED cases carried
    ``worst_rel = 1.0``. The absolute number is the honest measure there.
    """
    ref = np.array([0.0, 1.1e-17, -4.0e-18])
    candidate = ref + 1.1e-17

    diff = compare_field(ref, candidate, "p_rgh")

    assert diff.matched is True
    assert diff.rel_diff == 0.0
    assert diff.abs_diff < ATOL


def test_a_zero_reference_still_reports_a_candidate_that_diverged() -> None:
    """The near-zero guard must not swallow a real blow-up.

    A reference that is exactly zero with a candidate at 1e5 has no relative scale
    either — but it is emphatically not a match, so ``rel`` falls back to the
    absolute difference and keeps ``_classify`` reporting FIELDS_DIFFER rather than
    MATCHED_TO_ROUNDOFF.
    """
    ref = np.zeros(3)
    candidate = np.array([0.0, 1.0e5, 0.0])

    diff = compare_field(ref, candidate, "p_rgh")

    assert diff.matched is False
    assert diff.rel_diff == 1.0e5
    assert _classify([diff])[0] == FIELDS_DIFFER
