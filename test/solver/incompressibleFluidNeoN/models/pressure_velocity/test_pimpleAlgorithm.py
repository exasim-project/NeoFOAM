# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Dictionary reads the NeoN PIMPLE build performs, tested without a case.

``pimpleAlgorithm`` decides three things by reading the *converted* (NeoN)
``system/fvSolution``: which control block the case ships (``PIMPLE`` or
``PISO``), the corrector counts in it, and the momentumPredictor switch. Those
reads are pure functions of a ``NeoN::Dictionary`` — no mesh, no ``Foam::Time``,
no Kokkos initialization — so they are exercised here in-process instead of
through a solver run.

The dictionaries are built with the NeoN API rather than read from a case:
NeoFOAM exposes no Python-level OpenFOAM->NeoN dictionary conversion, so the
converted form is only reachable through a full runtime. The entries mirror the
blocks the cases here run key for key — ``test/setup_pimple``'s ``PIMPLE``, the
``PISO`` block ``test_control_block_selection`` patches in its place, and
``cases/regexSolverKeys``' regex-keyed one — and the on-disk counterpart stays
covered by ``test_control_block_selection``, which initializes a real pisoFoam
case in a subprocess.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import neon._neon as nn  # NeoN Python bindings
import pytest

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.pimpleAlgorithm import (
    _control_block_name,
    _read_int,
    _read_switch,
    pimpleNeoN,
)


def _fv_solution(*control_blocks: str) -> Any:
    """An fvSolution dictionary carrying the named (empty) control blocks."""
    fv_solution = nn.Dictionary()
    for block in control_blocks:
        fv_solution.insert_dict(block, nn.Dictionary())
    return fv_solution


@pytest.mark.parametrize(
    ("control_blocks", "expected"),
    [
        (("PIMPLE",), "PIMPLE"),
        (("PISO",), "PISO"),
        (("PIMPLE", "PISO"), "PIMPLE"),
    ],
    ids=["pimpleFoam case", "pisoFoam case", "both blocks"],
)
def test_control_block_name_selects_the_block_the_case_ships(
    control_blocks: tuple[str, ...], expected: str
) -> None:
    """PIMPLE wins where both are written; a PISO-only case selects PISO."""
    fv_solution = _fv_solution(*control_blocks)

    assert _control_block_name(fv_solution) == expected


def test_control_block_name_without_either_block_raises() -> None:
    """A case with no control block is named as such, not left to fail in NeoN.

    The unguarded ``subDict("PIMPLE")`` this replaced raised NeoN's
    ``Key 'PIMPLE' not found in Dictionary``, which names neither the file nor
    the alternative.
    """
    with pytest.raises(ValueError, match="neither a PIMPLE nor a PISO block"):
        _control_block_name(_fv_solution())


def test_pressure_reference_uses_the_selected_control_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pressure reference is read from the same block as the corrector counts.

    ``set_ref_cell`` looks ``pRefCell``/``pRefValue`` up in the block it is
    handed, so a hardcoded ``"PIMPLE"`` aborted a pisoFoam case with a FOAM
    fatal *Entry 'PIMPLE' not found*. The step is driven through the spec's real
    build steps; only ``set_ref_cell`` (which needs a runtime and a mesh) is
    stubbed, so the block name it receives is the observation.
    """
    passed: list[str] = []

    def record_algorithm(runtime: Any, field: str, algorithm: str) -> tuple[int, float, bool]:
        passed.append(algorithm)
        return (0, 0.0, False)

    monkeypatch.setattr(nfb, "set_ref_cell", record_algorithm)
    step = next(s for s in pimpleNeoN.build_steps() if s.name == "models.pressure_reference")

    step.initializer({"_neon_runtime": SimpleNamespace(fv_solution_dict=_fv_solution("PISO"))})

    assert passed == ["PISO"]


def test_read_int_reads_the_entry() -> None:
    """A written corrector count is returned instead of the default."""
    control = nn.Dictionary()
    control.insert_int("nCorrectors", 2)

    assert _read_int(control, "nCorrectors", 1) == 2


def test_read_int_falls_back_to_the_default() -> None:
    """An absent key yields the OpenFOAM default the caller passes."""
    assert _read_int(nn.Dictionary(), "nCorrectors", 1) == 1


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("yes", True),
        ("no", False),
        ("on", True),
        ("off", False),
        ("true", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("  Yes  ", True),
    ],
)
def test_read_switch_reads_an_openfoam_switch_word(written: str, expected: bool) -> None:
    """Every switch spelling OpenFOAM accepts maps to the same boolean."""
    control = nn.Dictionary()
    control.insert_string("momentumPredictor", written)

    # The default is the opposite of the expectation, so a value that was not
    # read would show up as the wrong answer rather than the right one.
    assert _read_switch(control, "momentumPredictor", not expected) is expected


def test_read_switch_falls_back_to_the_default() -> None:
    """An absent switch yields the OpenFOAM default the caller passes."""
    assert _read_switch(nn.Dictionary(), "momentumPredictor", True) is True


def test_read_switch_reads_a_switch_a_typed_getter_rejects() -> None:
    """Switch words arrive as strings, so the read must not go through get_bool.

    The OpenFOAM->NeoN conversion stores ``momentumPredictor no;`` as the word
    ``"no"``; asking the dictionary for a bool instead fails the ``any_cast``.
    """
    control = nn.Dictionary()
    control.insert_string("momentumPredictor", "no")
    with pytest.raises(RuntimeError, match="bad any_cast"):
        control.get_bool("momentumPredictor")

    assert _read_switch(control, "momentumPredictor", True) is False
