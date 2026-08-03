# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoF under decomposition: damBreak on two ranks.

Everything the serial suite proves about this solver is blind to one class of
mistake — a reduction that is only *locally* correct.
``compute_alpha_courant_number`` (a ``gMax`` over the near-interface flux sum
and a ``gSum`` over that sum and over the cell volumes), the adaptive ``deltaT``
derived from it, the closed-domain pressure reference and the MULES limiter all
collapse to plain array arithmetic on one rank, so a missing MPI reduction is
invisible in serial and fatal in parallel. This module is the first coverage of
that path.

**Oracle.** The one the serial damBreak comparison uses, run on two ranks: the
identical decomposed case is advanced by native ``interFoam -parallel`` and by
``incompressibleVoF -parallel``, both reconstructed, and the fields are required
to agree to ``rtol = atol = 1e-10``. Both solvers share the CFL primitives and
therefore the adaptive-dt sequence, so machine precision — not a modelling
tolerance — is the bar. Measured, the two agree far better than that: at
0.05 s, ``max|Δalpha.water| = 2e-79`` and ``U``/``p_rgh`` are bit-identical.

**Decomposition independence.** Decomposing a case is *not* expected to leave
the answer bit-identical: the DIC preconditioner of the ``p_rgh`` PCG solve is
block-local, so a two-rank run converges to a different point inside the same
``tolerance 1e-7`` ball. The honest bar is therefore native ``interFoam``'s own
serial-to-parallel spread on this case, which is measured in the same session
and used as the limit. Measured at 0.05 s the two solvers' spreads are equal to
every digit printed — ``max|Δ|`` of 4.8e-05 (alpha.water), 7.7e-03 (U) and
4.6e-01 (p_rgh) for *both* interFoam and incompressibleVoF — i.e. the Python
solver reacts to decomposition exactly as the reference implementation does.

**Reductions, read off the solver's own output.** ``set_time_step`` prints the
interface Courant numbers and the chosen ``deltaT`` every step. The *mean*
interface Courant number never feeds back into the solution — it is printed and
nothing else — so its ``gSum`` is pinned here and nowhere else in the suite; and
with the tutorial's ``maxAlphaCo 1`` the *max* never sets dt either, which is why
one regime lowers it until it does. Both numbers are compared step by step
against interFoam's own ``alphaCourantNo.H`` line on the same decomposition, and
their serial-to-parallel drift is held to interFoam's.

**Decomposition.** Two subdomains split along y (the tutorial's own
``system/decomposeParDict`` is replaced), so both ranks own part of the water
column and both develop interface cells — see ``_parallel_decomposeParDict``.
An x-split would leave the whole interface on rank 0 for this horizon, and a
rank-local ``gMax`` would still give the right answer.

**Horizon.** ``endTime = 0.05 s`` — 13 adaptive steps (171 in the
``maxAlphaCo 0.02`` regime), the horizon ``test_damBreak_comparison.py`` and
``test_mules_regimes.py`` also use, and long enough for the collapsing column to
spread interface cells over both ranks (at t=0 the ``setFields`` interface is
sharp, so the near-interface band ``0.01 <= alpha <= 0.99`` is empty and every
alpha-Courant number is 0). Four solver runs per regime, four regimes, ~22 s in
total.

Every run is its own process: ``run(...)`` constructs a ``Foam::Time`` and a
process may own exactly one.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pybFoam import dictionary

from ..incompressibleFluid.comparison_helpers import (
    compare_solver_fields,
    get_time_directories,
    read_internal_fields,
    setup_case,
)
from .comparison_helpers import FIELDS_TO_COMPARE, write_alpha_controls
from .parallel_helpers import (
    NPROCS,
    decompose,
    install_decompose_par_dict,
    rank_case,
    reconstruct,
    run_mpi,
    run_parallel_solver,
)

AlphaControls = Mapping[str, Union[bool, float]]
TimeControls = Mapping[str, float]
# (system/fvSolution alpha controls, system/controlDict entries)
Regime = Tuple[AlphaControls, TimeControls]

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_TUTORIAL = _REPO_ROOT / "tutorials" / "damBreak"
_SERIAL_WORKER = Path(__file__).parent / "_mules_regime_worker.py"

_END_TIME = 0.05
_WRITE_INTERVAL = 0.05

# Each entry is (system/fvSolution alpha controls, system/controlDict entries),
# written into all four cases of the regime with the OpenFOAM format's own
# reader/writer. Data, not code — TEST_STYLE rule 10.
#
# * the tutorial defaults;
# * the two alpha-advection branches whose parallel behaviour differs
#   structurally — the explicit MULES solve, and the sub-cycled loop, which
#   re-derives the flux and re-runs the limiter once per sub-step so every
#   reduction inside MULES happens twice as often;
# * ``maxAlphaCo 0.02``, the regime in which the *interface* Courant number
#   actually sets the time step. With the tutorial's ``maxAlphaCo 1`` it never
#   does — alphaCoNum peaks at 0.06 and dt is capped by the +20%/step ramp — so
#   without this entry no field comparison here would depend on the alpha-CFL
#   ``gMax`` at all (verified: a rank-local max leaves every field and every dt
#   in the other three regimes untouched).
REGIMES = [
    pytest.param(({}, {}), id="tutorial_defaults"),
    pytest.param(({"MULESCorr": False}, {}), id="MULESCorr_off"),
    pytest.param(({"nAlphaSubCycles": 2}, {}), id="nAlphaSubCycles_2"),
    pytest.param(({}, {"maxAlphaCo": 0.02}), id="maxAlphaCo_0p02"),
]

# ``Interface Courant Number mean: <m>  max: <M>`` and ``deltaT = <dt>``, the
# two lines ``set_time_step`` prints per step. interFoam's alphaCourantNo.H
# prints the first in the same words (with one space), so the same pattern reads
# both logs.
_COURANT_LINE = re.compile(r"Interface Courant Number mean: (\S+)\s+max: (\S+)")
_DELTA_T_LINE = re.compile(r"^deltaT = (\S+)", re.MULTILINE)

# incompressibleVoF vs interFoam on the *same* decomposition: same algorithm,
# same dt sequence, so machine precision.
_SOLVER_RTOL = 1e-10

# Agreement bar for the printed Courant numbers, and the floor under the
# serial-to-parallel drift checks. This solver prints them with ``%.4g``, so two
# runs holding the same number can still straddle a rounding boundary by ~1e-4
# relative; measured against interFoam's ``%.6g`` line on the same decomposition
# the worst case is 3.4e-4. A reduction that dropped a rank moves these by O(1)
# — measured 1.0 relative for both the gMax and the gSum — three orders up.
_PRINTED_RTOL = 1e-3

# Slack on "no worse than interFoam's own serial-to-parallel spread": the two
# spreads came out equal to every printed digit, so this only absorbs a
# last-bit difference from a different MPI/BLAS build.
_SPREAD_SLACK = 1.05


@dataclass
class DamBreakRuns:
    """One regime, run four ways; the cases stay on disk for the assertions."""

    regime: Regime
    parallel_python: Path
    parallel_native: Path
    serial_python: Path
    serial_native: Path
    parallel_log: str
    serial_log: str
    parallel_native_log: str
    serial_native_log: str


def _describe(regime: Regime) -> str:
    """The regime's dictionary overrides, for assertion messages."""
    alpha_controls, time_controls = regime
    return f"fvSolution={dict(alpha_controls)} controlDict={dict(time_controls)}"


def _write_time_controls(control_dict_path: Path, entries: Mapping[str, float]) -> None:
    """Overwrite top-level ``system/controlDict`` entries in place.

    Through pybFoam's own ``dictionary.read``/``set``/``write`` — the OpenFOAM
    format's reader and writer — so the test never patches dictionary text
    (TEST_STYLE rule 3), exactly as ``write_alpha_controls`` does for fvSolution.
    """
    control_dict = dictionary.read(str(control_dict_path))
    for key, value in entries.items():
        control_dict.set(key, value)
    control_dict.write(str(control_dict_path))


def _prepare(case: Path, regime: Regime) -> None:
    alpha_controls, time_controls = regime
    setup_case(_TUTORIAL, case, _END_TIME, _WRITE_INTERVAL, run_setfields=True)
    if alpha_controls:
        write_alpha_controls(case / "system" / "fvSolution", alpha_controls)
    if time_controls:
        _write_time_controls(case / "system" / "controlDict", time_controls)


def _run_serial_solver(case: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(_SERIAL_WORKER), str(case)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"serial incompressibleVoF failed in {case}:\n"
        f"{result.stdout[-6000:]}\n{result.stderr[-4000:]}"
    )
    return result.stdout


def _run_serial_native(case: Path) -> str:
    result = subprocess.run(
        ["interFoam", "-case", str(case)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"serial interFoam failed:\n{result.stderr[-4000:]}"
    return result.stdout


@pytest.fixture(scope="module", params=REGIMES)
def dam_break_runs(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> DamBreakRuns:
    """Run one regime serially and on two ranks, with both solvers."""
    regime: Regime = request.param
    root = tmp_path_factory.mktemp("damBreakParallel")
    cases = {
        name: root / name
        for name in (
            "parallel_python",
            "parallel_native",
            "serial_python",
            "serial_native",
        )
    }
    for case in cases.values():
        _prepare(case, regime)

    for case in (cases["parallel_python"], cases["parallel_native"]):
        install_decompose_par_dict(case)
        decompose(case)

    parallel_log = run_parallel_solver(cases["parallel_python"])
    parallel_native_log = run_mpi(["interFoam", "-parallel"], cases["parallel_native"])
    reconstruct(cases["parallel_python"])
    reconstruct(cases["parallel_native"])

    serial_log = _run_serial_solver(cases["serial_python"])
    serial_native_log = _run_serial_native(cases["serial_native"])

    return DamBreakRuns(
        regime=regime,
        parallel_log=parallel_log,
        parallel_native_log=parallel_native_log,
        serial_log=serial_log,
        serial_native_log=serial_native_log,
        **cases,
    )


def _final_fields(case: Path) -> dict[str, np.ndarray]:
    names = [name for name, _ in FIELDS_TO_COMPARE]
    return read_internal_fields(case, get_time_directories(case)[-1], names)


# --------------------------------------------------------------------------- #
# The fields                                                                   #
# --------------------------------------------------------------------------- #


def test_parallel_damBreak_matches_native_interFoam(
    dam_break_runs: DamBreakRuns,
) -> None:
    """Two ranks, same decomposition, same dictionaries: bit-for-bit interFoam."""
    all_match, failed_fields, failed_details = compare_solver_fields(
        dam_break_runs.parallel_python,
        dam_break_runs.parallel_native,
        FIELDS_TO_COMPARE,
        rtol=_SOLVER_RTOL,
        atol=_SOLVER_RTOL,
    )

    assert all_match, f"regime {_describe(dam_break_runs.regime)} on 2 ranks: " + ", ".join(
        f"{name}(abs={failed_details[name][0]:.3e}, rel={failed_details[name][1]:.3e})"
        for name in failed_fields
    )


def test_decomposing_the_case_moves_the_answer_no_more_than_it_moves_interFoam(
    dam_break_runs: DamBreakRuns,
) -> None:
    """Decomposition independence, held to the reference implementation's own bar.

    Serial and parallel cannot agree bit-for-bit (block-local DIC), so the
    limit per field is what native interFoam does to itself between the same two
    runs, measured in this same session rather than hard-coded.
    """
    python_serial = _final_fields(dam_break_runs.serial_python)
    python_parallel = _final_fields(dam_break_runs.parallel_python)
    native_serial = _final_fields(dam_break_runs.serial_native)
    native_parallel = _final_fields(dam_break_runs.parallel_native)

    for field_name, _field_type in FIELDS_TO_COMPARE:
        python_spread = float(
            np.max(np.abs(python_parallel[field_name] - python_serial[field_name]))
        )
        native_spread = float(
            np.max(np.abs(native_parallel[field_name] - native_serial[field_name]))
        )
        assert python_spread <= _SPREAD_SLACK * native_spread, (
            f"regime {_describe(dam_break_runs.regime)}: {field_name} moves by "
            f"{python_spread:.3e} between the serial and the 2-rank "
            f"incompressibleVoF run, more than the {native_spread:.3e} native "
            "interFoam moves between the same two runs"
        )


# --------------------------------------------------------------------------- #
# The reductions, as the solver itself reports them                            #
# --------------------------------------------------------------------------- #


def _interface_courant_numbers(log: str) -> list[tuple[float, float]]:
    return [(float(mean), float(peak)) for mean, peak in _COURANT_LINE.findall(log)]


def _time_steps(log: str) -> list[float]:
    return [float(value) for value in _DELTA_T_LINE.findall(log)]


def test_parallel_interface_courant_numbers_match_native_interFoam(
    dam_break_runs: DamBreakRuns,
) -> None:
    """The Python ``gMax``/``gSum`` composition reproduces OpenFOAM's own
    ``alphaCourantNo.H`` step by step on the same two ranks.

    The *mean* is the reason this test exists: it is computed with ``pyf.sum``
    (pybFoam exposes ``Foam::gSum`` under that name) and then only printed, so
    no other assertion in the suite would notice if it were a rank-local sum.
    Both ranks own interface cells, so a local sum reports a visibly different
    number.
    """
    parallel = _interface_courant_numbers(dam_break_runs.parallel_log)
    native = _interface_courant_numbers(dam_break_runs.parallel_native_log)

    assert len(parallel) == len(native) > 1, (
        f"regime {_describe(dam_break_runs.regime)}: {len(parallel)} steps from "
        f"incompressibleVoF vs {len(native)} from interFoam"
    )
    assert_allclose(
        parallel,
        native,
        rtol=_PRINTED_RTOL,
        atol=0,
        err_msg=f"damBreak {_describe(dam_break_runs.regime)} on 2 ranks: interface "
        "Courant numbers (mean, max) differ from native interFoam",
    )


def _max_relative_drift(first: Sequence[Any], second: Sequence[Any]) -> float:
    """Largest elementwise relative difference between two printed sequences."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    assert a.shape == b.shape and a.size > 1, (
        f"the two runs printed different sequences: {a.shape} vs {b.shape}"
    )
    scale = np.maximum(np.abs(a), np.abs(b))
    difference = np.abs(a - b)
    return float(np.max(np.where(scale > 0, difference / np.where(scale > 0, scale, 1), 0.0)))


def test_decomposing_the_case_moves_the_interface_courant_numbers_no_more_than_interFoam(
    dam_break_runs: DamBreakRuns,
) -> None:
    """Serial and 2-rank do not print *identical* Courant numbers, and cannot:
    the p_rgh solve is preconditioner-dependent, so phi differs in the last
    digits and — once dt is alpha-CFL-limited — the difference compounds step by
    step (measured 3.4e-3 over the 171 steps of the ``maxAlphaCo 0.02`` regime,
    1.5e-4 over the 13 steps of the others). The meaningful bar is again what
    native interFoam does to itself between the same two runs; the floor is this
    solver's own ``%.4g`` print granularity, which alone puts ~1e-3 between two
    runs holding the same number.
    """
    python_drift = _max_relative_drift(
        _interface_courant_numbers(dam_break_runs.parallel_log),
        _interface_courant_numbers(dam_break_runs.serial_log),
    )
    native_drift = _max_relative_drift(
        _interface_courant_numbers(dam_break_runs.parallel_native_log),
        _interface_courant_numbers(dam_break_runs.serial_native_log),
    )
    limit = max(_SPREAD_SLACK * native_drift, _PRINTED_RTOL)

    assert python_drift <= limit, (
        f"regime {_describe(dam_break_runs.regime)}: the interface Courant "
        f"numbers move by {python_drift:.3e} between the serial and the 2-rank "
        f"incompressibleVoF run, past the {limit:.3e} allowed by native "
        f"interFoam's own {native_drift:.3e}"
    )


def test_decomposing_the_case_moves_the_time_step_sequence_no_more_than_interFoam(
    dam_break_runs: DamBreakRuns,
) -> None:
    """Every rank must pick the same dt, or the ranks desynchronise — and the dt
    a decomposed run picks must track the serial one.

    Held to interFoam's own serial-to-parallel drift for the same reason as the
    Courant numbers. Note this compares *this solver's* two logs only: the
    printed value is the dt before ``Time::adjustDeltaT`` snaps it to the write
    interval, where interFoam prints the value after, so the two solvers' lines
    are not the same quantity even though the dt they run with is (which the
    machine-precision field comparison above is what proves).
    """
    python_drift = _max_relative_drift(
        _time_steps(dam_break_runs.parallel_log),
        _time_steps(dam_break_runs.serial_log),
    )
    native_drift = _max_relative_drift(
        _time_steps(dam_break_runs.parallel_native_log),
        _time_steps(dam_break_runs.serial_native_log),
    )
    limit = max(_SPREAD_SLACK * native_drift, _PRINTED_RTOL)

    assert python_drift <= limit, (
        f"regime {_describe(dam_break_runs.regime)}: the adaptive deltaT sequence "
        f"moves by {python_drift:.3e} between the serial and the 2-rank run, past "
        f"the {limit:.3e} allowed by native interFoam's own {native_drift:.3e}"
    )


# --------------------------------------------------------------------------- #
# The run really was decomposed                                                #
# --------------------------------------------------------------------------- #


def test_parallel_run_writes_one_time_directory_per_rank(
    dam_break_runs: DamBreakRuns,
) -> None:
    """Each rank wrote its own share of the fields — the comparisons above are
    not two serial runs in disguise."""
    for rank in range(NPROCS):
        written = dam_break_runs.parallel_python / f"processor{rank}" / "0.05"
        assert written.is_dir(), f"rank {rank} wrote no 0.05 directory"
        assert (written / "alpha.water").is_file()


def test_the_decomposition_puts_interface_cells_on_both_ranks(
    dam_break_runs: DamBreakRuns, tmp_path: Path
) -> None:
    """Anchors every reduction claim above: if one rank never saw the interface,
    a rank-local ``gMax`` on the other one would satisfy them all."""
    for rank in range(NPROCS):
        case = rank_case(dam_break_runs.parallel_python, rank, tmp_path / f"r{rank}")
        alpha = read_internal_fields(case, case / "0.05", ["alpha.water"])["alpha.water"]
        band = int(np.count_nonzero((alpha >= 0.01) & (alpha <= 0.99)))
        assert band > 0, f"rank {rank} holds no near-interface cell at t = 0.05"
