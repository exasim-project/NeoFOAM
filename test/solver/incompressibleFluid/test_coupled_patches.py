# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The fluid as a region of a multi-region case, and its coupled patch of ``T``.

``cases/heatedChannel``: region ``fluid/`` only (1 x 0.3, 4 x 3 cells, Boussinesq,
laminar), ``T`` 300 K inside and 310 K on the wall ``interface`` at y = 0. Its
``system/controlDict`` has times of its own (endTime 5, deltaT 1); the region
steps on those of the case's ``system/controlDict.yaml`` (0.02, 0.01).

The region is initialized uncoupled; :class:`CoupledTemperature` is built on it
with ``Coupling.add`` stubbed (adding a side needs a coupled case: the coupled
run is ``test_heated_plate.py``).
"""

import gzip
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pytest

from neofoam.framework.coupling import Coupling
from neofoam.solver.incompressibleFluid.coupled_patches import (
    CoupledTemperature,
    _boussinesq_config,
    coupled_entries,
    coupled_patches,
)
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid
from neofoam.solver.incompressibleFluid.models.field_writer import (
    CoupledEntriesWriteHook,
    keep_coupled_entries,
    restore_coupled_entries,
)

CASE = Path(__file__).parent / "cases" / "heatedChannel"
RHO_CP, NU, PR = 5000.0, 2e-4, 0.01
DELTA_COEFF = 1.0 / 0.05  # the wall cells are 0.1 high


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
def patch(region: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Iterator[CoupledTemperature]:
    """A coupled patch on ``interface``, its 0/T values put back after the test."""
    added: list[Any] = []
    monkeypatch.setattr(Coupling, "add", lambda self, side: added.append(side))
    ctx = region["ctx"]
    side = CoupledTemperature("T", "interface", ctx)
    assert added == [side]
    yield side
    side.set_wall_T(np.full(4, 310.0))
    np.asarray(ctx.fields["alphat"]["interface"])[:] = 0.0


# --- the region reads its own case directory ---------------------------------


def test_the_region_reads_its_case_directory(region: dict[str, Any]) -> None:
    ctx = region["ctx"]
    assert ctx.region == "fluid"
    assert ctx.fields["T"].mesh().nCells() == 12  # fluid/constant/polyMesh
    np.testing.assert_allclose(np.asarray(ctx.fields["T"]["interface"]), 310.0)
    # fluid/system/fvSolution, fluid/constant/transportProperties
    assert ctx.models["pressure_velocity"].config.case_dir == Path("fluid")
    config = _boussinesq_config(ctx)
    assert config is not None and (config.rho, config.cp) == (1.0, 5000.0)
    # no constant/ or 0/ at the case's root
    assert not (region["case_dir"] / "constant").exists()


def test_the_region_steps_on_the_times_of_the_case(region: dict[str, Any]) -> None:
    loops = [
        model
        for model in region["runtime"].state.core_models
        if getattr(getattr(model, "spec", None), "name", None) in ("solutionLoop", "fieldWriter")
    ]
    assert len(loops) == 2
    for model in loops:  # system/controlDict.yaml, not fluid/system/controlDict
        assert (model.config.endTime, model.config.deltaT) == (0.02, 0.01)


def test_an_uncoupled_region_has_no_coupled_patches(region: dict[str, Any]) -> None:
    ctx = region["ctx"]
    assert ctx.models["coupled_patches"] == []
    assert type(ctx.models["writer"].hook).__name__ == "RuntimeWriteHook"


# --- the coupled patch ---------------------------------------------------------


def test_the_points_are_the_face_centres(patch: CoupledTemperature) -> None:
    points = np.asarray(patch.points())
    np.testing.assert_allclose(points[:, 0], [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(points[:, 1:], [[0.0, 0.05]] * 4)


def test_the_received_wall_temperature_is_the_fixed_value(
    patch: CoupledTemperature, region: dict[str, Any]
) -> None:
    T = region["ctx"].fields["T"]
    values = np.array([301.0, 302.0, 303.0, 304.0])
    patch.receive(patch.__coupled__.receives(T=values))
    np.testing.assert_array_equal(np.asarray(T["interface"]), values)
    T.correctBoundaryConditions()  # a fixedValue patch keeps them
    np.testing.assert_array_equal(patch.wall_T(), values)


def test_the_heat_flux_into_the_fluid(patch: CoupledTemperature, region: dict[str, Any]) -> None:
    # rho cp (nu/Pr + alphat) (T_wall - T_cell) deltaCoeffs: 5000 * 0.02 * 10 * 20
    sent = patch.send()
    np.testing.assert_allclose(sent.q, RHO_CP * NU / PR * 10.0 * DELTA_COEFF, rtol=1e-12)
    assert np.all(sent.q > 0)  # the wall is the hotter: into the fluid
    patch.set_wall_T([290.0, 300.0, 305.0, 310.0])
    np.asarray(region["ctx"].fields["alphat"]["interface"])[:] = 0.01
    np.testing.assert_allclose(
        patch.heat_flux(),
        RHO_CP * (NU / PR + 0.01) * np.array([-10.0, 0.0, 5.0, 10.0]) * DELTA_COEFF,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("unset", "message"), [(("rho",), "set rho "), (("rho", "cp"), "set rho and cp ")]
)
def test_rho_and_cp_are_required(
    region: dict[str, Any], monkeypatch: pytest.MonkeyPatch, unset: tuple[str, ...], message: str
) -> None:
    ctx = region["ctx"]
    monkeypatch.setattr(Coupling, "add", lambda self, side: None)
    runtime = next(
        m for m in ctx.models.values() if getattr(m, "config", None) is _boussinesq_config(ctx)
    )
    monkeypatch.setattr(runtime, "config", runtime.config.model_copy(update=dict.fromkeys(unset)))
    with pytest.raises(ValueError, match=message + r".*constant/transportProperties"):
        CoupledTemperature("T", "interface", ctx)


def test_the_boussinesq_model_is_required() -> None:
    ctx = SimpleNamespace(models={}, fields={})
    with pytest.raises(ValueError, match="needs the boussinesq model"):
        CoupledTemperature("T", "interface", ctx)  # type: ignore[arg-type]


# --- the boundary defines the coupling ------------------------------------------

_T = """FoamFile {{ version 2.0; format ascii; class volScalarField; object T; }}
dimensions [0 0 0 1 0 0 0]; internalField uniform 300;
boundaryField {{
 inlet {{ type fixedValue; value uniform 300; }}
 interface {{ type {type}; coupled {coupled}; value uniform 310; }}
 top {{ type zeroGradient; }}
}}
"""


def test_the_coupled_entries_of_a_field_file(tmp_path: Path) -> None:
    path = tmp_path / "T"
    path.write_text(_T.format(type="fixedValue", coupled="coupledTemperature"))
    assert coupled_entries("T", path) == {"interface": "coupledTemperature"}


@pytest.mark.parametrize(
    ("patch_type", "coupled", "message"),
    [
        ("fixedValue", "coupledHeatFlux", "coupled as 'coupledHeatFlux'; known"),
        ("calculated", "coupledTemperature", "has type calculated; .* is 'fixedValue'"),
    ],
)
def test_a_coupled_entry_is_checked(
    tmp_path: Path, patch_type: str, coupled: str, message: str
) -> None:
    path = tmp_path / "T"
    path.write_text(_T.format(type=patch_type, coupled=coupled))
    with pytest.raises(ValueError, match=message):
        coupled_entries("T", path)


def test_a_case_without_the_field_file_has_no_coupled_patches(tmp_path: Path) -> None:
    ctx = SimpleNamespace(models={"solution_loop": SimpleNamespace(timeName=lambda: "0")})
    assert coupled_patches("T", ctx, tmp_path) == []  # type: ignore[arg-type]


# --- the writer adds the coupled entries back --------------------------------------

#: as OpenFOAM's fixedValue writes a patch: no ``coupled`` entry
_WRITTEN = b"""FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}

dimensions      [0 0 0 1 0 0 0];

internalField   uniform 300;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform 300;
    }
    interface
    {
        type            fixedValue;
        value           nonuniform List<scalar> 2(301.5 302.5);
    }
    interfaceToo
    {
        type            zeroGradient;
    }
}
"""


def test_the_coupled_entry_is_added_back(tmp_path: Path) -> None:
    path = tmp_path / "T"
    path.write_bytes(_WRITTEN)
    restore_coupled_entries(path, {"interface": "coupledTemperature"})
    assert coupled_entries("T", path) == {"interface": "coupledTemperature"}
    # only that patch, the rest of the file as written
    assert (
        path.read_bytes().replace(b"        coupled         coupledTemperature;\n", b"") == _WRITTEN
    )
    restore_coupled_entries(path, {"interface": "coupledTemperature"})  # once
    assert path.read_bytes().count(b"coupled ") == 1


def test_the_coupled_entry_is_added_back_to_a_compressed_file(tmp_path: Path) -> None:
    (tmp_path / "T.gz").write_bytes(gzip.compress(_WRITTEN))
    restore_coupled_entries(tmp_path / "T", {"interface": "coupledTemperature"})
    assert not (tmp_path / "T").exists()
    assert (
        b"    {\n        coupled         coupledTemperature;\n"
        b"        type            fixedValue;\n"
        b"        value           nonuniform" in gzip.decompress((tmp_path / "T.gz").read_bytes())
    )


def test_a_patch_the_written_file_lacks_raises(tmp_path: Path) -> None:
    path = tmp_path / "T"
    path.write_bytes(_WRITTEN)
    with pytest.raises(ValueError, match="no patch wall in boundaryField"):
        restore_coupled_entries(path, {"wall": "coupledTemperature"})


def test_the_writer_hook_adds_them_after_each_write(tmp_path: Path) -> None:
    class _Inner:
        """Writes T as OpenFOAM does, at the current time."""

        def __init__(self) -> None:
            self.time = "0.5"

        def write_fields(self, fields: Any) -> None:
            (tmp_path / "fluid" / self.time).mkdir(parents=True)
            (tmp_path / "fluid" / self.time / "T").write_bytes(_WRITTEN)

    inner = _Inner()
    writer = SimpleNamespace(hook=inner)
    keep_coupled_entries(writer, None, tmp_path / "fluid", {"T": {}})
    assert writer.hook is inner  # nothing coupled, nothing changes
    time = SimpleNamespace(timeName=lambda: inner.time)
    keep_coupled_entries(
        writer, time, tmp_path / "fluid", {"T": {"interface": "coupledTemperature"}}
    )
    assert isinstance(writer.hook, CoupledEntriesWriteHook) and writer.hook.inner is inner
    for written in ("0.5", "1"):
        inner.time = written
        writer.hook.write_fields({})
        assert coupled_entries("T", tmp_path / "fluid" / written / "T") == {
            "interface": "coupledTemperature"
        }
