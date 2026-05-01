# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Test that StagedInit.run() fails fast when fvSchemes/fvSolution requirements
are not met. Uses a minimal solver wired through the real StagedInit.
"""

from typing import Any

import pytest

from neofoam.foam import fvSchemes, fvSolution
from neofoam.foam.fv_configs import FvSchemesConfig, FvSolutionConfig
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import InitStep, LoadResult, StagedInit
from neofoam.framework.model import Model


# ============================================================================
# Minimal model with @fvSchemes.add / @fvSolution.add
# ============================================================================


solver_model = Model("test_solver")


@solver_model.load
def load(_case_dir: Any, _entry: Any) -> None:
    return None


@solver_model.operation(operation_number="1")
@fvSchemes.add(ddt="default", div="div(phi,U)", grad="default")
@fvSolution.add("U")
def momentum(self: Any) -> FieldUpdates:
    return FieldUpdates({})


@solver_model.operation(operation_number="2", depends_on=["momentum"])
@fvSchemes.add(laplacian="default")
@fvSolution.add("p")
def pressure(self: Any) -> FieldUpdates:
    return FieldUpdates({})


# ============================================================================
# Helper: build a StagedInit with fv configs from dicts
# ============================================================================


def _make_init(
    fv_schemes_data: dict[str, Any],
    fv_solution_data: dict[str, Any],
) -> StagedInit:
    """Wire a StagedInit that loads fvSchemes/fvSolution from dicts."""
    init = StagedInit("test_solver")

    @init.load
    def load_config() -> LoadResult:
        fv_schemes = FvSchemesConfig.model_construct(**fv_schemes_data)
        fv_solution = FvSolutionConfig.model_construct(**fv_solution_data)
        return LoadResult(
            core_models=[solver_model],
            optional_models=[],
            fv_schemes_config=fv_schemes,
            fv_solution_config=fv_solution,
        )

    @init.resolve
    def resolve(config: Any) -> None:
        pass

    @init.build
    def build(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        return []

    return init


# ============================================================================
# Tests
# ============================================================================


VALID_SCHEMES = {
    "ddtSchemes": {"default": "Euler"},
    "divSchemes": {"div(phi,U)": "Gauss linear"},
    "gradSchemes": {"default": "Gauss linear"},
    "laplacianSchemes": {"default": "Gauss linear corrected"},
}

VALID_SOLUTION = {"solvers": {"U": {}, "p": {}}}


def test_valid_case_passes() -> None:
    init = _make_init(VALID_SCHEMES, VALID_SOLUTION)
    ctx = init.run()
    assert ctx is not None


def test_missing_scheme_fails_fast() -> None:
    schemes_no_ddt = {
        "divSchemes": {"div(phi,U)": "Gauss linear"},
        "gradSchemes": {"default": "Gauss linear"},
        "laplacianSchemes": {"default": "Gauss linear corrected"},
    }
    init = _make_init(schemes_no_ddt, VALID_SOLUTION)
    with pytest.raises(RuntimeError, match="ddtSchemes.default"):
        init.run()


def test_missing_solver_fails_fast() -> None:
    solution_no_p = {"solvers": {"U": {}}}
    init = _make_init(VALID_SCHEMES, solution_no_p)
    with pytest.raises(RuntimeError, match="solvers.p"):
        init.run()


def test_multiple_errors_collected() -> None:
    schemes_empty = {"divSchemes": {}}
    solution_empty = {"solvers": {}}
    init = _make_init(schemes_empty, solution_empty)
    with pytest.raises(RuntimeError) as exc_info:
        init.run()
    msg = str(exc_info.value)
    assert "ddtSchemes.default" in msg
    assert "div(phi,U)" in msg
    assert "solvers.U" in msg
    assert "solvers.p" in msg


def test_no_fv_configs_skips_verification() -> None:
    """When fv_schemes_config / fv_solution_config are None, verification is skipped."""
    init = StagedInit("test_no_fv")

    @init.load
    def load_config() -> LoadResult:
        return LoadResult(core_models=[], optional_models=[])

    @init.resolve
    def resolve(config: Any) -> None:
        pass

    @init.build
    def build(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        return []

    ctx = init.run()
    assert ctx is not None
