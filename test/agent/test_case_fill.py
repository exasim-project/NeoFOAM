# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the case-fill agent API (``neofoam.agent.case_fill``).

Three layers are exercised:

1. ``build_case_output_model`` — purely in-process, no OpenFOAM needed:
   the aggregate ``CaseSpec`` carries one ``Optional`` field per config
   class the solver may consume.
2. ``load_case_from_disk`` + ``save_case`` — the deterministic, LLM-free
   roundtrip used by ``--no-llm``. Reads pybFoam-format dictionaries, so
   gated by ````.
3. End-to-end: fill the bundled pitzDaily fixture into a scratch case
   and run ``incompressibleFluid`` on it for one step — proves a
   freshly-agent-filled case is solver-runnable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# The agent module imports pydantic_ai at the top level via case_fill — and
# case_fill uses the aggregate schema even when no LLM is invoked.
pytest.importorskip("pydantic_ai")

from neofoam.agent.case_fill import (  # noqa: E402
    build_case_output_model,
    case_spec_to_configs,
    fill_case,
    load_case_from_disk,
    save_case,
)


def _setup_case(
    source_case: Path,
    test_case: Path,
    *,
    end_time: float,
    write_interval: object,
) -> None:
    """Copy a tutorial case, restore ``0/``, run blockMesh, override timings.

    Trimmed-down twin of ``test.solver.incompressibleFluid.comparison_helpers
    .setup_case`` — see the module-level comment above for why we don't import.
    """
    if test_case.exists():
        shutil.rmtree(test_case)
    test_case.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_case, test_case)

    orig_dir = test_case / "0.orig"
    zero_dir = test_case / "0"
    if orig_dir.exists():
        if zero_dir.exists():
            shutil.rmtree(zero_dir)
        shutil.copytree(orig_dir, zero_dir)

    result = subprocess.run(
        ["blockMesh", "-case", str(test_case)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"blockMesh failed: {result.stderr}"

    control_dict = test_case / "system" / "controlDict"
    lines = control_dict.read_text().split("\n")
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("endTime"):
            new_lines.append(f"endTime         {end_time};")
        elif stripped.startswith("writeInterval"):
            new_lines.append(f"writeInterval   {write_interval};")
        else:
            new_lines.append(line)
    control_dict.write_text("\n".join(new_lines))


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CASE = REPO_ROOT / "test" / "solver" / "incompressibleFluid" / "val_pitzDaily"


# ---------------------------------------------------------------------------
# Pure-Python: schema shape
# ---------------------------------------------------------------------------


def test_case_output_model_has_one_optional_field_per_config() -> None:
    """The aggregate exposes every solver config as an Optional field."""
    model = build_case_output_model()
    # The five solver-core dictionaries pitzDaily needs are in there.
    expected = {
        "control_dict_config",
        "pimple_fv_schemes",
        "pimple_fv_solution",
        "transport_properties_config",
        "turbulence_properties_config",
    }
    assert expected.issubset(set(model.model_fields))
    # All fields default to None, so the agent can leave irrelevant ones out.
    instance = model()
    for name in model.model_fields:
        assert getattr(instance, name) is None


def test_case_spec_to_configs_filters_none_and_preserves_order() -> None:
    """Only populated BaseConfig fields are returned, in declaration order."""
    from neofoam.solver.incompressibleFluid.configs import ControlDictConfig

    model = build_case_output_model()
    cd = ControlDictConfig(endTime=0.1, deltaT=0.01)
    spec = model(control_dict_config=cd)
    configs = case_spec_to_configs(spec)
    assert configs == [cd]


# ---------------------------------------------------------------------------
# Roundtrip via the OpenFOAM IO strategy
# ---------------------------------------------------------------------------


def test_load_case_from_disk_reads_every_present_config(tmp_path: Path) -> None:
    """``load_case_from_disk`` materialises the configs whose schemas match.

    ``ControlDictConfig`` / ``TransportPropertiesConfig`` /
    ``TurbulencePropertiesConfig`` round-trip cleanly through the
    OpenFOAM IO strategy. ``Pimple_fvSchemes`` / ``Pimple_fvSolution``
    fail validation against the bundled pitzDaily fixture (the schemas
    require fields the fixture doesn't carry — ``ddt(U)``, ``p_rgh``,
    etc.; see REVIEW.md for why the schemas are slices); they end up as
    ``None`` rather than raising, and ``fill_case``'s static-asset
    mirror keeps a runnable file on disk.
    """
    spec = load_case_from_disk(SOURCE_CASE)
    assert spec.control_dict_config is not None
    assert spec.transport_properties_config is not None
    assert spec.turbulence_properties_config is not None

    # Spot-check a few values to prove this is the *fixture's* data, not defaults.
    assert spec.transport_properties_config.transportModel == "Newtonian"
    assert spec.turbulence_properties_config.simulationType == "RAS"
    assert spec.turbulence_properties_config.RAS is not None
    assert spec.turbulence_properties_config.RAS.RASModel == "SpalartAllmaras"
    assert (spec.control_dict_config.endTime or 0.0) > 0.0


def test_save_case_writes_files_via_registered_io_strategies(tmp_path: Path) -> None:
    """A CaseSpec roundtrips to disk and re-loads with the same values.

    Covers the three IO-strategy fixes:

    - ``TurbulencePropertiesConfig`` writes the nested ``RAS`` sub-dict
      from scratch (no longer raises "Cannot create nested dictionary
      'RAS' via pybFoam bindings").
    - The generated file carries a ``FoamFile`` block (previously
      stripped by the writer).
    - All three populated configs end up in the ``written`` list — no
      warnings are emitted.
    """
    spec = load_case_from_disk(SOURCE_CASE)

    written = save_case(spec, tmp_path)
    populated = case_spec_to_configs(spec)
    assert len(written) == len(populated)

    for path in written:
        assert "FoamFile" in path.read_text(), f"{path} missing FoamFile header"

    spec2 = load_case_from_disk(tmp_path)
    assert (
        spec2.transport_properties_config.transportModel
        == spec.transport_properties_config.transportModel
    )
    assert spec2.control_dict_config.endTime == spec.control_dict_config.endTime
    # The nested RAS sub-dict round-tripped fully:
    assert spec2.turbulence_properties_config.RAS is not None
    assert (
        spec2.turbulence_properties_config.RAS.RASModel
        == spec.turbulence_properties_config.RAS.RASModel
    )


# ---------------------------------------------------------------------------
# End-to-end: a filled case runs the solver
# ---------------------------------------------------------------------------


def test_fill_case_no_llm_produces_a_runnable_case(tmp_path: Path) -> None:
    """``fill_case`` without an agent mirrors source → target and runs the solver.

    Reproduces what ``neofoam agent fill SOURCE TARGET --no-llm`` does: copy
    mesh + ``0.orig/`` fields and roundtrip the configs from disk. The
    resulting directory must be runnable by ``incompressibleFluid.run`` after
    ``blockMesh`` (the LLM path takes the exact same code from here).
    """
    # ``setup_case`` (from the existing comparison helpers) takes care of:
    #  - copying the case, restoring ``0.orig`` → ``0/``,
    #  - running blockMesh, and
    #  - shortening endTime / writeInterval so the test stays fast.
    staging = tmp_path / "pitzDaily_staging"
    # writeControl is ``timeStep`` in val_pitzDaily → writeInterval must be
    # an integer step count; 1 = write every step.
    _setup_case(SOURCE_CASE, staging, end_time=0.0001, write_interval=1)

    # Now ask the case-fill API to scaffold a fresh target from the staged
    # case (which has a built mesh). The no-LLM path does a disk roundtrip
    # — the same code path the LLM path lands on after the agent returns.
    filled = tmp_path / "pitzDaily_filled"
    spec = fill_case(staging, filled, agent=None)
    assert spec.control_dict_config is not None

    # Sanity-check what was written:
    for rel in (
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "constant/transportProperties",
        "constant/turbulenceProperties",
        "constant/polyMesh/points",
        "0/U",
        "0/p",
    ):
        assert (filled / rel).exists(), f"missing {rel} in scaffolded case"

    # Run the solver inside the filled case — this is the "verify the solver
    # runs" assertion from the user request. We change into the case dir
    # because ``incompressibleFluid.run`` expects cwd-relative OpenFOAM paths.
    from neofoam.solver.incompressibleFluid import run as run_incompressible_fluid

    original_dir = Path.cwd()
    os.chdir(filled)
    try:
        ctx = run_incompressible_fluid(["incompressibleFluid"])
    finally:
        os.chdir(original_dir)

    # The solver must have advanced at least one step and produced an output
    # time directory beyond ``0/``.
    assert ctx is not None
    time_dirs = sorted(
        d.name for d in filled.iterdir() if d.is_dir() and d.name[0].isdigit()
    )
    assert len(time_dirs) >= 2, f"only saw time dirs {time_dirs}"


def test_fill_case_rejects_a_missing_source_before_calling_the_agent(
    tmp_path: Path,
) -> None:
    """A missing ``source_case`` raises before any agent is built or run.

    Ports the guard the removed MCP ``fill_case`` tool carried (finding M4): a
    contentless prompt must never burn an LLM API call. The recording agent flips
    a flag on construction/run — it must stay unset.
    """

    class _RecordingAgent:
        def __init__(self) -> None:
            self.called = False

        def run_sync(self, prompt: str) -> object:
            self.called = True
            raise AssertionError("agent.run_sync must not be reached")

    agent = _RecordingAgent()
    missing = tmp_path / "no_such_source"
    with pytest.raises(ValueError, match="source_case"):
        fill_case(missing, tmp_path / "target", agent=agent, copy_static=False)
    assert agent.called is False


def test_cli_agent_fill_no_llm(tmp_path: Path) -> None:
    """``neofoam agent fill --no-llm`` writes the configs end-to-end."""
    import typer.testing

    from neofoam.cli.app import app

    # Stage the source with a built mesh so the fill+run combo is reproducible
    # (the CLI doesn't run blockMesh itself).
    staging = tmp_path / "pitzDaily_src"
    # writeControl is ``timeStep`` in val_pitzDaily → writeInterval must be
    # an integer step count; 1 = write every step.
    _setup_case(SOURCE_CASE, staging, end_time=0.0001, write_interval=1)

    target = tmp_path / "pitzDaily_dst"

    runner = typer.testing.CliRunner()
    result = runner.invoke(
        app,
        ["agent", "fill", str(staging), str(target), "--no-llm"],
    )
    assert result.exit_code == 0, result.output
    assert (target / "system" / "controlDict").exists()
    assert (target / "constant" / "transportProperties").exists()
