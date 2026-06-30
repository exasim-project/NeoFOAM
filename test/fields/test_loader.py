# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Full ``load_fields`` / ``save_fields`` round-trip on a real OpenFOAM case.

The fixture is a copy of the upstream
``tutorials/heatTransfer/buoyantBoussinesqPimpleFoam/hotRoom/0.orig``
directory under ``test/fields/cases/`` (per the project's
``feedback_tests_use_real_files`` preference: dict-reading tests load
real OpenFOAM dictionaries from disk, not encoded strings).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from neofoam.fields.bc import FixedValueBC, GenericBC, NoSlipBC, ZeroGradientBC
from neofoam.fields.loader import load_fields, save_fields
from neofoam.fields.value_types import Scalar, Vector
from neofoam.framework.model.spec import Model


FIXTURE = Path(__file__).parent / "cases" / "hotRoom" / "0.orig"


class _SyntheticSolver:
    """Same duck-type used in ``test_configurations.py``."""

    def __init__(self, name: str, model_specs: list[object]) -> None:
        self.name = name
        self._config_classes: list[type] = []
        self.model_specs = model_specs


def _hot_room_solver() -> _SyntheticSolver:
    """Declare exactly the fields the hotRoom fixture carries on disk.

    Splitting them across a "pimple-shaped" and a "boussinesq-shaped"
    spec mirrors the future in-tree split (Step 8) — at this layer all
    the loader cares about is that ``solver.model_specs`` yields
    specs whose ``_field_decls`` cover the union.
    """
    pimple = Model("pimple")
    pimple.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC, FixedValueBC, GenericBC],
        write=True,
    )
    pimple.field(
        "p",
        dimensions=[0, 2, -2, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[FixedValueBC, ZeroGradientBC, GenericBC],
        write=True,
    )

    boussinesq = Model("boussinesq")
    boussinesq.field(
        "p_rgh",
        dimensions=[0, 2, -2, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[FixedValueBC, GenericBC],
        write=True,
    )
    boussinesq.field(
        "T",
        dimensions=[0, 0, 0, 1, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[FixedValueBC, ZeroGradientBC, GenericBC],
        write=True,
    )
    boussinesq.field(
        "alphat",
        dimensions=[0, 2, -1, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[FixedValueBC, GenericBC],
    )

    turbulence = Model("kEpsilon")
    turbulence.field(
        "k",
        dimensions=[0, 2, -2, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[FixedValueBC, GenericBC],
    )
    turbulence.field(
        "epsilon",
        dimensions=[0, 2, -3, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[FixedValueBC, GenericBC],
    )
    turbulence.field(
        "nut",
        dimensions=[0, 2, -1, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[FixedValueBC, GenericBC],
    )

    return _SyntheticSolver("hotRoom", [pimple, boussinesq, turbulence])


# -- helpers -----------------------------------------------------------


def _stage(tmp_path: Path) -> Path:
    """Copy ``0.orig/*`` into ``tmp_path/0/`` for a writable case."""
    case = tmp_path / "case"
    (case / "0").mkdir(parents=True)
    for f in FIXTURE.iterdir():
        shutil.copy(f, case / "0" / f.name)
    return case


# -- tests -------------------------------------------------------------


def test_load_fields_covers_every_declared_name(tmp_path: Path) -> None:
    case = _stage(tmp_path)
    fields = load_fields(case, solver=_hot_room_solver())
    assert set(fields) == {"U", "p", "p_rgh", "T", "alphat", "k", "epsilon", "nut"}


def test_loaded_dimensions_match_disk(tmp_path: Path) -> None:
    case = _stage(tmp_path)
    fields = load_fields(case, solver=_hot_room_solver())
    assert fields["U"].dimensions == [0, 1, -1, 0, 0, 0, 0]
    assert fields["T"].dimensions == [0, 0, 0, 1, 0, 0, 0]
    assert fields["p_rgh"].dimensions == [0, 2, -2, 0, 0, 0, 0]


def test_typed_arm_dispatch(tmp_path: Path) -> None:
    """U has only noSlip BCs in the fixture — the typed arm wins."""
    case = _stage(tmp_path)
    fields = load_fields(case, solver=_hot_room_solver())
    for patch, bc in fields["U"].boundaryField.items():
        assert isinstance(bc, NoSlipBC), (patch, bc)


def test_generic_fallback_keeps_unknown_bcs_intact(tmp_path: Path) -> None:
    """p_rgh uses fixedFluxPressure with extras; GenericBC preserves them."""
    case = _stage(tmp_path)
    fields = load_fields(case, solver=_hot_room_solver())
    for patch, bc in fields["p_rgh"].boundaryField.items():
        assert isinstance(bc, GenericBC), (patch, bc)
        dumped = bc.model_dump()
        assert dumped["type"] == "fixedFluxPressure"
        assert dumped["rho"] == "rhok"


def test_round_trip_preserves_payload(tmp_path: Path) -> None:
    """load_fields → save_fields → reload should produce the same dump."""
    case = _stage(tmp_path)
    solver = _hot_room_solver()

    before = load_fields(case, solver=solver)
    save_fields(before, case)
    after = load_fields(case, solver=solver)

    for name in before:
        assert before[name].model_dump() == after[name].model_dump(), name


def test_mutation_before_save_is_persisted(tmp_path: Path) -> None:
    case = _stage(tmp_path)
    solver = _hot_room_solver()

    fields = load_fields(case, solver=solver)
    # Flip the floor BC on T from fixedValue → zeroGradient; reload and assert.
    fields["T"].boundaryField["floor"] = ZeroGradientBC()
    save_fields({"T": fields["T"]}, case)

    reloaded = load_fields(case, solver=solver)
    assert isinstance(reloaded["T"].boundaryField["floor"], ZeroGradientBC)
    # The other patches survive untouched.
    assert isinstance(reloaded["T"].boundaryField["ceiling"], FixedValueBC)


def test_save_fields_rejects_unbound_instance(tmp_path: Path) -> None:
    case = _stage(tmp_path)
    from pydantic import BaseModel

    class _Unbound(BaseModel):
        pass

    with pytest.raises(ValueError, match="no io_config"):
        save_fields({"x": _Unbound()}, case)  # type: ignore[dict-item]
