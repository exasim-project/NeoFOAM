# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Tests for solver_inputs() and scheme_inputs() — config discovery at init time.

solver_inputs() → model config classes (what parameters each model accepts)
scheme_inputs() → typed fvSchemes model (what discretization schemes the solver needs)

Both work at import time — no load() or run() needed.
"""

import pytest
from pydantic import ValidationError

from neofoam.io import BaseConfig
from neofoam.framework.initialization import StagedInit

from .dummy_solver import CoreAlgorithmConfig, ScalarTransportConfig, WallModelConfig
from .dummy_solver.core_algorithm import core_algorithm
from .dummy_solver.solver import init as dummy_init

# ============================================================================
# solver_inputs() — "What parameters does each model accept?"
# ============================================================================


def test_solver_inputs() -> None:
    """Discovers config classes from core models and plugin models."""
    inputs = dummy_init.solver_inputs()

    # Core model (registered via register_core_models)
    assert inputs["core_algorithm"] is CoreAlgorithmConfig

    # Plugin models (registered via .register_with)
    assert inputs["wall_model"] is WallModelConfig
    assert inputs["scalar_transport"] is ScalarTransportConfig

    for name, cls in inputs.items():
        assert issubclass(cls, BaseConfig), f"{name} is not a BaseConfig"


def test_solver_inputs_core_only() -> None:
    """Without plugin_interface, only core models are visible."""
    init = StagedInit("core_only")
    init.register_core_models([core_algorithm])
    inputs = init.solver_inputs()
    assert "core_algorithm" in inputs
    assert "wall_model" not in inputs


def test_solver_inputs_empty() -> None:
    assert StagedInit("empty").solver_inputs() == {}


# ============================================================================
# scheme_inputs() — "What fvSchemes entries does the solver need?"
# ============================================================================


def test_scheme_inputs() -> None:
    """Builds typed model with concrete keys from all registered operations."""
    Model = dummy_init.scheme_inputs()
    props = list(Model.model_json_schema()["properties"].keys())

    # All 6 standard scheme sections + wallDist covered
    assert any("ddtSchemes" in p for p in props)
    assert any("divSchemes" in p for p in props)
    assert any("gradSchemes" in p for p in props)
    assert any("laplacianSchemes" in p for p in props)
    assert any("snGradSchemes" in p for p in props)
    assert any("interpolationSchemes" in p for p in props)
    assert any("wallDist" in p for p in props)

    # Concrete keys, not "default"
    assert "ddtSchemes_ddt_U" in props
    assert "divSchemes_div_phi_U" in props
    assert "gradSchemes_grad_p" in props
    assert "wallDist_method" in props
    assert "divSchemes_div_phi_T" in props  # from scalar_transport

    # Each field has a description
    for prop in Model.model_json_schema()["properties"].values():
        assert "description" in prop


def test_scheme_inputs_validates() -> None:
    """Valid values parse, invalid values reject."""
    Model = dummy_init.scheme_inputs()

    # Valid
    instance = Model(
        ddtSchemes_ddt_U="Euler",
        ddtSchemes_ddt_nuTilda="Euler",
        ddtSchemes_ddt_T="Euler",
        divSchemes_div_phi_U="Gauss upwind",
        divSchemes_div_phi_nuTilda="Gauss upwind",
        divSchemes_div_phi_T="Gauss upwind",
        gradSchemes_grad_U="Gauss linear",
        gradSchemes_grad_p="Gauss linear",
        gradSchemes_grad_nuTilda="Gauss linear",
        laplacianSchemes_laplacian_nuEff_U="Gauss linear corrected",
        laplacianSchemes_laplacian_rAU_p="Gauss linear corrected",
        laplacianSchemes_laplacian_DnuTildaEff_nuTilda="Gauss linear corrected",
        laplacianSchemes_laplacian_alphaEff_T="Gauss linear corrected",
        interpolationSchemes_flux_HbyA="linear",
        interpolationSchemes_interpolate_rAU="linear",
        snGradSchemes_snGrad_p="corrected",
        wallDist_method="meshWave",
    )
    assert instance.ddtSchemes_ddt_U.type == "Euler"  # type: ignore[union-attr]

    # Invalid
    with pytest.raises(ValidationError):
        Model(ddtSchemes_ddt_U="invalidScheme")


def test_scheme_inputs_empty() -> None:
    Model = StagedInit("empty").scheme_inputs()
    assert Model.model_json_schema()["properties"] == {}


# ============================================================================
# Stage 2: real incompressibleFluid solver
# ============================================================================


def test_real_solver_inputs() -> None:
    """Real solver discovers all model configs without load()."""
    from neofoam.solver.incompressibleFluid.create_fields import init as real_init

    inputs = real_init.solver_inputs()

    # Core models (Piso is a stub with no config yet)
    assert "Pimple" in inputs
    assert "Simple" in inputs

    # Plugin models
    assert "boussinesq" in inputs
    assert "spalart_allmaras" in inputs

    for name, cls in inputs.items():
        assert issubclass(cls, BaseConfig), f"{name} is not a BaseConfig"


def test_real_solver_scheme_inputs() -> None:
    """Real solver builds typed scheme model without load()."""
    from neofoam.solver.incompressibleFluid.create_fields import init as real_init

    Model = real_init.scheme_inputs()
    props = list(Model.model_json_schema()["properties"].keys())

    # From pimple momentum
    assert any("ddtSchemes" in p for p in props)
    assert any("divSchemes" in p for p in props)
    # From SA turbulence
    assert any("wallDist" in p for p in props)
