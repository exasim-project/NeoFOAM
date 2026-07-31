# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Diff two finished runs field by field.

Field reads go through :func:`neofoam.tooling.casebuild.read_field`, which spawns
one interpreter per read: constructing ``Foam::Time`` twice in one process
corrupts OpenFOAM's per-process global state, and an in-process second read
silently returns the *previous* case's fields — a bug that turns every
comparison into a false pass.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

from neofoam.tooling.casebuild import CaseDir
from verification.dropin.execute import (
    COMPARE_FAILED,
    FIELDS_DIFFER,
    MATCHED,
    MATCHED_TO_ROUNDOFF,
    MESH_DIFFERS,
    MESH_NOT_REPRODUCIBLE,
    mesh_fingerprint,
    time_dirs,
)

__all__ = ["FieldDiff", "compare_field", "compare_runs"]

#: Strict on purpose. The native and neofoam solvers assemble the same discrete
#: system, so a correct port reproduces it to round-off; a looser tolerance would
#: hide a real algorithmic difference behind "close enough".
RTOL = 1e-10
ATOL = 1e-15

#: A differing field below this relative disagreement reproduces native to machine
#: precision (:data:`MATCHED_TO_ROUNDOFF`), not a defect. Set to ``RTOL``: a field can
#: fail the *combined* ``rtol``/``atol`` test yet still be this close in relative terms
#: when its peak magnitude pushes one ULP above the tiny ``atol``.
ROUNDOFF = 1e-10


class FieldDiff:
    """One field's worst absolute and relative disagreement between two runs."""

    def __init__(self, name: str, matched: bool, abs_diff: float, rel_diff: float):
        self.name = name
        self.matched = matched
        self.abs_diff = abs_diff
        self.rel_diff = rel_diff

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "matched": self.matched,
            "abs": self.abs_diff,
            "rel": self.rel_diff,
        }


def compare_field(reference: np.ndarray, candidate: np.ndarray, name: str) -> FieldDiff:
    """Compare two internal fields elementwise against ``RTOL``/``ATOL``.

    ``rel`` is normalized by the reference field's peak magnitude, so a reader can
    tell round-off in near-zero entries (``rel`` ~ 1e-13, numerically equivalent)
    from genuine divergence (``rel`` ~ 1e-1) even though both fail the strict
    ``atol``.

    That normalization is only meaningful while the reference field *has* a scale.
    A field that is numerically zero everywhere (``peak <= ATOL`` — e.g. ``p_rgh``
    on a case that never develops pressure) has no relative error to speak of, and
    dividing one round-off by another yields an arbitrary ratio: it reported
    ``rel = 1.0`` for a ``p_rgh`` matching to 1e-17, which says the opposite of the
    truth. With no reference scale the *absolute* difference is the only honest
    measure, so it is used directly — zero when the two agree to ``ATOL`` (both
    numerically zero), and the raw difference when they do not, which keeps a
    candidate that diverged from a zero reference reading as a real disagreement
    rather than a match.
    """
    if reference.shape != candidate.shape:
        return FieldDiff(name, False, float("inf"), float("inf"))
    abs_diff = float(np.max(np.abs(candidate - reference)))
    peak = float(np.max(np.abs(reference)))
    if peak > ATOL:
        rel_diff = abs_diff / peak
    else:
        rel_diff = 0.0 if abs_diff <= ATOL else abs_diff
    matched = bool(np.allclose(candidate, reference, rtol=RTOL, atol=ATOL))
    return FieldDiff(name, matched, abs_diff, rel_diff)


def _classify(diffs: list[FieldDiff]) -> tuple[str, str]:
    """Fold per-field diffs into one outcome + detail.

    Order matters. A field whose two internal arrays have different lengths is a
    *mesh* difference (AMR changed the final-time cell count), reported as
    :data:`MESH_DIFFERS` rather than an ``inf`` :data:`FIELDS_DIFFER` that reads as a
    numerical blow-up. Otherwise: all within strict tolerance ⇒ :data:`MATCHED`; the
    remaining disagreements all below :data:`ROUNDOFF` ⇒ :data:`MATCHED_TO_ROUNDOFF`;
    anything larger ⇒ :data:`FIELDS_DIFFER`.
    """
    shape_mismatch = [d.name for d in diffs if math.isinf(d.rel_diff)]
    if shape_mismatch:
        detail = "internal field length differs (mesh changed): " + ", ".join(shape_mismatch)
        return MESH_DIFFERS, detail
    bad = [d for d in diffs if not d.matched]
    if not bad:
        return MATCHED, ""
    detail = ", ".join(d.name for d in bad)
    if all(d.rel_diff < ROUNDOFF for d in bad):
        return MATCHED_TO_ROUNDOFF, detail
    return FIELDS_DIFFER, detail


#: How much of a failed read's message to keep in ``detail``. Enough for an OpenFOAM
#: fatal (message + file + line + the function it came from), which is what a reader
#: failure looks like; the message is the only record, `COMPARE_FAILED` has no log.
_DETAIL_CHARS = 700


def _read_failure(failure: Exception) -> str:
    """The *tail* of a failed read's message — the reader's stderr, not its argv.

    The head is the command that was run, which says nothing; keeping the last line
    only is just as useless (an OpenFOAM fatal ends on ``FOAM exiting``).
    """
    lines = [line.strip() for line in str(failure).splitlines() if line.strip()]
    return "field read failed: " + " / ".join(lines)[-_DETAIL_CHARS:]


def _worse(existing: FieldDiff | None, candidate: FieldDiff) -> FieldDiff:
    """The larger-disagreement of two diffs for the same field (across ranks)."""
    if existing is None or candidate.rel_diff > existing.rel_diff:
        return candidate
    return existing


def _is_decomposed(run_dir: Path) -> bool:
    """True when the run left its results only in ``processor*/`` (no reconstruct).

    Wave tutorials end at ``runParallel <solver>`` with no ``reconstructPar``, so the
    written times live under ``processor*/`` while the case root carries only the
    initial ``0/``. Detected by a processor dir reaching a *later* time than the root —
    so a serial run (no processors) and a reconstructed run (root has every time) both
    take the ordinary serial path unchanged.
    """
    procs = sorted(run_dir.glob("processor*"))
    if not procs:
        return False
    root_times = time_dirs(run_dir)
    proc_times = time_dirs(procs[0])
    root_latest = float(root_times[-1].name) if root_times else -1.0
    proc_latest = float(proc_times[-1].name) if proc_times else -1.0
    return proc_latest > root_latest


def _proc_index(proc: Path) -> int:
    return int(proc.name[len("processor") :])


def _read_decomposed_field(proc_dir: Path, system_src: Path, name: str, time: str) -> np.ndarray:
    """Read one rank's field by assembling a single-domain case from the processor.

    A ``processor*`` dir carries its own decomposed ``constant/polyMesh`` and time
    dirs but no ``system/`` — so a case built from the run root's ``system`` plus the
    processor's ``constant`` and time dir reads exactly that rank's cells through the
    ordinary subprocess reader.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp) / "case"
        shutil.copytree(system_src / "system", case / "system")
        shutil.copytree(proc_dir / "constant", case / "constant")
        shutil.copytree(proc_dir / time, case / time)
        return CaseDir(case).read_field(name, time=time)


def _compare_decomposed(
    native_dir: Path, neo_dir: Path, fields: list[str]
) -> tuple[str, str, list[FieldDiff]]:
    """Diff two decomposed runs rank-by-rank, aggregating the worst diff per field.

    Both sides ran the same decomposition, so rank *r* owns identical cells on each
    side — comparing ``processor*/<t>/<field>`` pairwise needs no ``reconstructPar``
    and no OpenFOAM utility. Each field's reported diff is its worst across all ranks.
    """
    native_procs = sorted(native_dir.glob("processor*"), key=_proc_index)
    neo_procs = sorted(neo_dir.glob("processor*"), key=_proc_index)
    if len(native_procs) != len(neo_procs):
        detail = f"processor count differs: native {len(native_procs)} vs neofoam {len(neo_procs)}"
        return COMPARE_FAILED, detail, []

    worst: dict[str, FieldDiff] = {}
    try:
        for native_proc, neo_proc in zip(native_procs, neo_procs):
            native_times = time_dirs(native_proc)
            neo_times = time_dirs(neo_proc)
            if not native_times or not neo_times:
                return COMPARE_FAILED, "a processor wrote no time directory", []
            native_time, neo_time = native_times[-1], neo_times[-1]
            if native_time.name != neo_time.name:
                detail = (
                    f"latest time differs on {native_proc.name}: native "
                    f"{native_time.name} vs neofoam {neo_time.name}"
                )
                return COMPARE_FAILED, detail, []
            for name in fields:
                if not (native_time / name).is_file():
                    continue
                if not (neo_time / name).is_file():
                    continue
                diff = compare_field(
                    _read_decomposed_field(native_proc, native_dir, name, native_time.name),
                    _read_decomposed_field(neo_proc, neo_dir, name, neo_time.name),
                    name,
                )
                worst[name] = _worse(worst.get(name), diff)
    except Exception as exc:  # a field that cannot be read is a compare fault
        return COMPARE_FAILED, _read_failure(exc), []

    if not worst:
        return COMPARE_FAILED, "no comparable fields written", []
    diffs = [worst[name] for name in fields if name in worst]
    outcome, detail = _classify(diffs)
    return outcome, detail, diffs


def compare_runs(
    native_dir: Path, neo_dir: Path, fields: list[str]
) -> tuple[str, str, list[FieldDiff]]:
    """Compare the final-time *fields* of two finished runs.

    Returns ``(outcome, detail, diffs)``. Guards that must pass before any field
    is trusted: the two meshes must be identical (else a nondeterministic mesher,
    not the solver, explains a difference — :data:`MESH_NOT_REPRODUCIBLE`, a
    harness fault distinct from the post-solve :data:`MESH_DIFFERS`), and the two
    runs must have reached the *same* final time (else they simulated different
    spans and are not comparable).

    A run left decomposed (``processor*/`` only, no ``reconstructPar`` — the wave
    tutorials) is diffed rank-by-rank; a serial or reconstructed run is diffed at the
    case root as before.
    """
    if mesh_fingerprint(native_dir) != mesh_fingerprint(neo_dir):
        return MESH_NOT_REPRODUCIBLE, "meshing is not reproducible; solver not comparable", []

    if _is_decomposed(native_dir) and _is_decomposed(neo_dir):
        return _compare_decomposed(native_dir, neo_dir, fields)

    native_times = time_dirs(native_dir)
    neo_times = time_dirs(neo_dir)
    if not native_times or not neo_times:
        return COMPARE_FAILED, "a run wrote no time directory", []
    native_time, neo_time = native_times[-1], neo_times[-1]
    if native_time.name != neo_time.name:
        detail = f"latest time differs: native {native_time.name} vs neofoam {neo_time.name}"
        return COMPARE_FAILED, detail, []

    present = [
        name for name in fields if (native_time / name).is_file() and (neo_time / name).is_file()
    ]
    if not present:
        return COMPARE_FAILED, "no comparable fields written", []

    native_case = CaseDir(native_dir)
    neo_case = CaseDir(neo_dir)
    try:
        diffs = [
            compare_field(
                native_case.read_field(name, time=native_time.name),
                neo_case.read_field(name, time=neo_time.name),
                name,
            )
            for name in present
        ]
    except Exception as exc:  # a field that cannot be read is a compare fault
        return COMPARE_FAILED, _read_failure(exc), []
    outcome, detail = _classify(diffs)
    return outcome, detail, diffs
