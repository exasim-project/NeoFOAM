# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import subprocess
import sys
from typing import Any

import pytest

from neofoam.io.schema import (
    ConfigInfo,
    ConfigSchema,
    ModelSummary,
    ToolInfo,
    _describe,
    config_schema,
    list_configs,
    model_catalog,
    tool_catalog,
)

pytest.importorskip("pybFoam")  # resolving the solver spec imports pybFoam


@pytest.fixture
def solver() -> Any:
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

    return incompressibleFluid


class _Documented:
    """First line is the purpose.

    Second paragraph is ignored by the describer.
    """


class _Undocumented:  # noqa: D101 - intentionally has no docstring
    __doc__ = None


def test_describe_returns_first_docstring_line() -> None:
    assert _describe(_Documented) == "First line is the purpose."


def test_describe_falls_back_when_no_docstring() -> None:
    # No getattr, no crash: a docstring-less object yields the fallback.
    assert _describe(_Undocumented, fallback="fallback text") == "fallback text"
    assert _describe(_Undocumented) is None


def test_list_configs_uses_snake_name_and_reports_file(solver: Any) -> None:
    configs = list_configs(solver)
    assert all(isinstance(c, ConfigInfo) for c in configs)
    cd = next(c for c in configs if c.cls_name == "ControlDictConfig")
    assert cd.name == "control_dict_config"
    assert cd.file == "system/controlDict"


def test_list_configs_classifies_origin(solver: Any) -> None:
    # origin tells a caller what it must author: controlDict is solver-declared
    # (always applies, not model-owned); transport/turbulence are owned by the
    # required Newtonian/laminar models; gravity is owned by the optional boussinesq.
    by_cls = {c.cls_name: c for c in list_configs(solver)}
    assert by_cls["ControlDictConfig"].origin == "solver"
    assert by_cls["TransportPropertiesConfig"].origin == "required_model"
    assert by_cls["TurbulencePropertiesConfig"].origin == "required_model"
    assert by_cls["GravityConfig"].origin == "optional_model"


def test_list_configs_neon_transport_is_solver_turbulence_is_model_owned() -> None:
    # After the turbulence-family merge incompressibleFluidNeoN binds the single
    # momentumTransportModel family (its members own turbulenceProperties), so
    # TurbulencePropertiesConfig is now required-model-owned — the same as the
    # pybFoam incompressibleFluid solver. Viscosity is still NOT a Python family on
    # NeoN (the C++ factory reads transportProperties directly), so
    # TransportPropertiesConfig stays solver-declared always-apply.
    from neofoam.solver.incompressibleFluidNeoN.incompressibleFluidNeoN import (
        incompressibleFluidNeoN,
    )

    by_cls = {c.cls_name: c for c in list_configs(incompressibleFluidNeoN)}
    assert by_cls["TransportPropertiesConfig"].origin == "solver"
    assert by_cls["TurbulencePropertiesConfig"].origin == "required_model"
    # transportProperties is not model-owned (no viscosity family), but
    # turbulenceProperties now is, so model_catalog lists it.
    owned = {d for e in model_catalog(incompressibleFluidNeoN) for d in e.dicts}
    assert "TransportPropertiesConfig" not in owned
    assert "TurbulencePropertiesConfig" in owned


def test_config_schema_returns_schema_ui_and_defaults(solver: Any) -> None:
    dto = config_schema(solver, "ControlDictConfig")
    assert isinstance(dto, ConfigSchema)
    assert dto.json_schema and "properties" in dto.json_schema
    assert dto.defaults["application"] == "pimpleFoam"
    assert isinstance(dto.ui_schema, dict)


def test_config_schema_accepts_snake_case_name(solver: Any) -> None:
    dto = config_schema(solver, "control_dict_config")
    assert dto.json_schema and "properties" in dto.json_schema


def test_config_schema_response_name_is_canonical_snake_case(solver: Any) -> None:
    # Either spelling in → the same canonical snake-case name out (no echo of input).
    from_class = config_schema(solver, "ControlDictConfig")
    from_snake = config_schema(solver, "control_dict_config")
    assert from_class.name == "control_dict_config"
    assert from_snake.name == "control_dict_config"


def test_config_schema_unknown_name_raises(solver: Any) -> None:
    with pytest.raises(ValueError):
        config_schema(solver, "NoSuchConfig")


def test_tool_catalog_links_tools_to_writing_config(solver: Any) -> None:
    catalog = tool_catalog(solver)
    by_name = {t.name: t for t in catalog}
    assert all(isinstance(t, ToolInfo) for t in catalog)
    assert {"blockMesh", "snappyHexMesh", "checkMesh"} <= set(by_name)
    block = by_name["blockMesh"]
    assert block.dict_file == "system/blockMeshDict"
    assert block.config == "BlockMeshDictConfig"
    assert by_name["snappyHexMesh"].config == "SnappyHexMeshDictConfig"
    assert by_name["checkMesh"].dict_file is None
    assert by_name["checkMesh"].config is None


def test_model_catalog_reports_required_and_optional(solver: Any) -> None:
    entries = model_catalog(solver)
    assert all(isinstance(e, ModelSummary) for e in entries)
    names = {e.name for e in entries}
    assert {"courant", "maxDeltaT", "boussinesq"} <= names
    required = {e.name for e in entries if e.required}
    assert {"Pimple", "Newtonian", "laminar"} <= required
    # owned configs serialize as class-name strings, never a class repr
    assert "ModelMetaclass" not in entries[0].model_dump_json()


def test_schema_module_imports_without_mcp() -> None:
    # io.schema is frontend-agnostic: importing it into a fresh interpreter pulls no
    # mcp / fastmcp (run in a subprocess so reloading its classes can't contaminate the
    # aliased mcp.dto types the rest of the suite isinstance-checks against). pybFoam is
    # a hard dependency the eager ``neofoam`` import always pulls, so it is not asserted.
    code = (
        "import sys, neofoam.io.schema\n"
        "assert not any(m.startswith('neofoam.mcp') for m in sys.modules), 'mcp leaked'\n"
        "assert 'fastmcp' not in sys.modules, 'fastmcp leaked'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
