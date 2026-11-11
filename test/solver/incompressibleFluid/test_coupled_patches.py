# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The fluid as a region of a multi-region case, and its coupled patch of ``T``.

``cases/heatedChannel``: region ``fluid/`` only (4 x 3 cells, Boussinesq), ``T``
300 K inside and 310 K on ``interface``; its own ``system/controlDict`` (endTime 5)
differs from the case's times (0.02). The region is initialized uncoupled; a
coupled patch is built without its binding: the coupled run is
``test_heated_plate.py``. Face
centres and heat fluxes are closed-form on this uniform mesh, so they hold to
round-off (``RTOL``).
"""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pytest

from neofoam.foam.coupling import FoamFields, FoamGeometry
from neofoam.framework.coupling import Cells, Patch
from neofoam.solver.incompressibleFluid.coupled_patches import (
    PATCH_TYPES,
    CoupledTemperature,
    _boussinesq_config,
    start_entries,
)
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid
from neofoam.solver.regions import WallTemperature, coupled_entries

CASE = Path(__file__).parent / "cases" / "heatedChannel"
ENTRIES = Path(__file__).parents[1] / "cases" / "coupledEntries"
RHO_CP, NU, PR = 5000.0, 2e-4, 0.01
DELTA_COEFF = 1.0 / 0.05  # the wall cells are 0.1 high
RTOL = 1e-12


@pytest.fixture(scope="module")
def region(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """The region ``fluid`` of a meshed copy of the case, initialized from the case's root."""
    case_dir = tmp_path_factory.mktemp("heatedChannel") / "case"
    shutil.copytree(CASE, case_dir)
    subprocess.run(
        ["blockMesh", "-case", str(case_dir / "fluid")],
        check=True,
        capture_output=True,
        timeout=120,
    )
    cwd = os.getcwd()
    os.chdir(case_dir)
    try:
        runtime = incompressibleFluid.instantiate(region="fluid")
        ctx = runtime.initialize()
        yield {"case_dir": case_dir, "runtime": runtime, "ctx": ctx}
    finally:
        os.chdir(cwd)


@pytest.fixture
def patch(region: dict[str, Any]) -> Iterator[CoupledTemperature]:
    """A coupled patch on ``interface``, its 0/T values put back after the test."""
    ctx = region["ctx"]
    instance = CoupledTemperature("T", "interface", ctx)
    yield instance
    instance.set_wall_T(np.full(4, 310.0))
    np.asarray(ctx.fields["alphat"]["interface"])[:] = 0.0


def test_fluid_region_reads_its_case_directory(region: dict[str, Any]) -> None:
    ctx = region["ctx"]
    assert ctx.region == "fluid"
    assert ctx.fields["T"].mesh().nCells() == 12  # fluid/constant/polyMesh
    np.testing.assert_allclose(
        np.asarray(ctx.fields["T"]["interface"]), 310.0, rtol=0, atol=0, err_msg="fluid/0/T"
    )
    # fluid/system/fvSolution, fluid/constant/transportProperties
    assert ctx.models["pressure_velocity"].config.case_dir == Path("fluid")
    config = _boussinesq_config(ctx)
    assert config is not None and (config.rho, config.cp) == (1.0, 5000.0)
    # no constant/ or 0/ at the case's root
    assert not (region["case_dir"] / "constant").exists()


def test_fluid_region_steps_on_the_times_of_the_case(region: dict[str, Any]) -> None:
    loops = [
        model
        for model in region["runtime"].state.core_models
        if getattr(getattr(model, "spec", None), "name", None) in ("solutionLoop", "fieldWriter")
    ]
    assert len(loops) == 2
    for model in loops:  # system/controlDict.yaml, not fluid/system/controlDict
        assert (model.config.endTime, model.config.deltaT) == (0.02, 0.01)


def test_fluid_region_uncoupled_has_no_coupled_patches(region: dict[str, Any]) -> None:
    ctx = region["ctx"]
    assert ctx.models["coupled_patches"] == []
    assert type(ctx.models["writer"].hook).__name__ == "RuntimeWriteHook"


def test_foam_geometry_gives_the_face_centres_of_a_patch(region: dict[str, Any]) -> None:
    points = FoamGeometry(region["ctx"].fields["T"].mesh()).face_centres("interface")

    np.testing.assert_allclose(
        points[:, 0], [0.125, 0.375, 0.625, 0.875], rtol=RTOL, err_msg="interface x"
    )
    np.testing.assert_allclose(points[:, 1:], [[0.0, 0.05]] * 4, atol=RTOL, err_msg="interface yz")


def test_coupled_temperature_receives_the_fixed_value(
    patch: CoupledTemperature, region: dict[str, Any]
) -> None:
    T = region["ctx"].fields["T"]
    values = np.array([301.0, 302.0, 303.0, 304.0])
    patch.receive(WallTemperature(T=values))
    np.testing.assert_array_equal(np.asarray(T["interface"]), values)
    T.correctBoundaryConditions()  # a fixedValue patch keeps them
    np.testing.assert_array_equal(patch.wall_T(), values)


def test_coupled_temperature_sends_the_heat_flux_into_the_fluid(
    patch: CoupledTemperature, region: dict[str, Any]
) -> None:
    # rho cp (nu/Pr + alphat) (T_wall - T_cell) deltaCoeffs: 5000 * 0.02 * 10 * 20
    sent = patch.send()
    np.testing.assert_allclose(
        sent.q, RHO_CP * NU / PR * 10.0 * DELTA_COEFF, rtol=RTOL, err_msg="initial q"
    )
    assert np.all(sent.q > 0)  # the wall is the hotter: into the fluid
    patch.set_wall_T([290.0, 300.0, 305.0, 310.0])
    np.asarray(region["ctx"].fields["alphat"]["interface"])[:] = 0.01
    np.testing.assert_allclose(
        patch.heat_flux(),
        RHO_CP * (NU / PR + 0.01) * np.array([-10.0, 0.0, 5.0, 10.0]) * DELTA_COEFF,
        rtol=RTOL,
        err_msg="q with alphat",
    )


@pytest.mark.parametrize(
    ("unset", "message"), [(("rho",), "set rho "), (("rho", "cp"), "set rho and cp ")]
)
def test_coupled_temperature_requires_rho_and_cp(
    region: dict[str, Any], monkeypatch: pytest.MonkeyPatch, unset: tuple[str, ...], message: str
) -> None:
    ctx = region["ctx"]
    runtime = next(
        m for m in ctx.models.values() if getattr(m, "config", None) is _boussinesq_config(ctx)
    )
    monkeypatch.setattr(runtime, "config", runtime.config.model_copy(update=dict.fromkeys(unset)))
    with pytest.raises(ValueError, match=message + r".*constant/transportProperties"):
        CoupledTemperature("T", "interface", ctx)


def test_coupled_temperature_requires_the_boussinesq_model() -> None:
    ctx = SimpleNamespace(models={}, fields={})
    with pytest.raises(ValueError, match="needs the boussinesq model"):
        CoupledTemperature("T", "interface", ctx)  # type: ignore[arg-type]


def test_coupled_entries_of_the_fluid_are_fixed_value_patches() -> None:
    T = Path(__file__).parent / "cases" / "heatedPlate" / "fluid" / "0" / "T"

    assert coupled_entries(T, PATCH_TYPES) == {"interface": "coupledTemperature"}


def test_coupled_entries_of_the_fluid_reject_a_calculated_patch() -> None:
    with pytest.raises(ValueError, match="has type calculated; a coupledTemperature patch is "):
        coupled_entries(ENTRIES / "coupled", PATCH_TYPES)


def test_start_entries_of_a_fluid_with_a_coupled_T_name_its_patches(tmp_path: Path) -> None:
    (tmp_path / "0").mkdir()
    shutil.copy(CASE.parent / "heatedPlate" / "fluid" / "0" / "T", tmp_path / "0" / "T")

    assert start_entries(tmp_path, "0") == {"interface": "coupledTemperature"}


def test_start_entries_of_a_fluid_without_T_are_none(tmp_path: Path) -> None:
    (tmp_path / "0").mkdir()

    assert start_entries(tmp_path, "0") == {}


def test_foam_fields_read_a_patch_and_the_cells(region: dict[str, Any]) -> None:
    fields = FoamFields(region["ctx"].fields)

    assert fields.read("T", Patch("interface")).tolist() == [310.0] * 4
    assert fields.read("T", Cells()).tolist() == [300.0] * 12


def test_foam_fields_write_a_patch_and_the_cells(region: dict[str, Any]) -> None:
    fields = FoamFields(region["ctx"].fields)

    fields.write("T", Patch("interface"), [301.0, 302.0, 303.0, 304.0])
    fields.write("T", Cells(), np.arange(12.0))

    written = (fields.read("T", Patch("interface")), fields.read("T", Cells()))
    fields.write("T", Patch("interface"), np.full(4, 310.0))
    fields.write("T", Cells(), np.full(12, 300.0))
    assert written[0].tolist() == [301.0, 302.0, 303.0, 304.0]
    assert written[1].tolist() == [float(cell) for cell in range(12)]
