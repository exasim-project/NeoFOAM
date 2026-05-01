# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for Spalart-Allmaras model plugin."""

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from neofoam.framework.model import ModelSpec
from neofoam.io import BaseConfig
from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
    spalart_allmaras,
)


def test_default_constants_match_standard_sa() -> None:
    """Standard SA constants from Spalart & Allmaras (1992)."""
    config = SpalartAllmarasConfig()
    assert config.Cb1 == 0.1355
    assert config.Cb2 == 0.622
    assert config.Cw2 == 0.3
    assert config.Cw3 == 2.0
    assert config.Cv1 == 7.1
    assert math.isclose(config.sigma, 2.0 / 3.0, rel_tol=1e-12)
    assert config.kappa == 0.41
    assert config.Cs == 0.3


def test_cw1_computed_from_other_constants() -> None:
    """Cw1 = Cb1 / kappa^2 + (1 + Cb2) / sigma."""
    config = SpalartAllmarasConfig()
    expected = config.Cb1 / config.kappa**2 + (1 + config.Cb2) / config.sigma
    assert math.isclose(config.Cw1, expected, rel_tol=1e-12)


def test_cw1_updates_with_custom_constants() -> None:
    """Cw1 is derived, so it changes when underlying constants change."""
    config = SpalartAllmarasConfig(Cb1=0.2, kappa=0.5, Cb2=0.5, sigma=1.0)
    expected = 0.2 / 0.5**2 + (1 + 0.5) / 1.0
    assert math.isclose(config.Cw1, expected, rel_tol=1e-12)


def test_config_is_baseconfig_subclass() -> None:
    """SpalartAllmarasConfig must be a BaseConfig subclass for IO integration."""
    config = SpalartAllmarasConfig()
    assert isinstance(config, BaseConfig)


# --- Cycle 2: Model registration and detection ---


def test_spalart_allmaras_is_model_spec() -> None:
    """The module-level spalart_allmaras must be a ModelSpec."""
    assert isinstance(spalart_allmaras, ModelSpec)
    assert spalart_allmaras.name == "spalart_allmaras"


def test_detect_true_for_sa() -> None:
    """detect returns True when turbulenceProperties has RASModel SpalartAllmaras."""
    mock_ras = MagicMock()
    mock_ras.found.return_value = True
    mock_ras.get_word.return_value = "SpalartAllmaras"

    mock_props = MagicMock()
    mock_props.found.return_value = True
    mock_props.subDict.return_value = mock_ras

    with patch(
        "neofoam.solver.incompressibleFluid.models.spalartAllmaras.pyf.dictionary.read",
        return_value=mock_props,
    ):
        result = spalart_allmaras.run_detect(Path("."))
        assert result.detected is True


def test_detect_false_for_kepsilon() -> None:
    """detect returns False when RASModel is kEpsilon."""
    mock_ras = MagicMock()
    mock_ras.found.return_value = True
    mock_ras.get_word.return_value = "kEpsilon"

    mock_props = MagicMock()
    mock_props.found.return_value = True
    mock_props.subDict.return_value = mock_ras

    with patch(
        "neofoam.solver.incompressibleFluid.models.spalartAllmaras.pyf.dictionary.read",
        return_value=mock_props,
    ):
        result = spalart_allmaras.run_detect(Path("."))
        assert result.detected is False


def test_detect_false_no_ras() -> None:
    """detect returns False when turbulenceProperties has no RAS subdict."""
    mock_props = MagicMock()
    mock_props.found.return_value = False

    with patch(
        "neofoam.solver.incompressibleFluid.models.spalartAllmaras.pyf.dictionary.read",
        return_value=mock_props,
    ):
        result = spalart_allmaras.run_detect(Path("."))
        assert result.detected is False


def test_detect_false_on_exception() -> None:
    """detect returns False if reading turbulenceProperties throws."""
    with patch(
        "neofoam.solver.incompressibleFluid.models.spalartAllmaras.pyf.dictionary.read",
        side_effect=RuntimeError("file not found"),
    ):
        result = spalart_allmaras.run_detect(Path("."))
        assert result.detected is False


# --- Cycle 3: Model loading ---


def test_load_returns_config() -> None:
    """Instantiating the model returns a runtime with SpalartAllmarasConfig."""
    entry: dict[str, Any] = {"type": "spalart_allmaras", "name": "spalart_allmaras"}
    rt = spalart_allmaras.instantiate(case_dir=Path("."), entry=entry)
    assert isinstance(rt.config, SpalartAllmarasConfig)
    assert rt.config.Cb1 == 0.1355
    assert rt.name == "spalart_allmaras"


# --- Cycle 4: Resolve ---


def test_resolve_is_registered() -> None:
    """SA resolve sets turbulence_type on the pressure-velocity algorithm."""
    assert spalart_allmaras._resolve_func is not None


def test_resolve_sets_turbulence_type() -> None:
    """Resolve sets turbulence_type='spalart_allmaras' on the algorithm model."""
    from neofoam.framework.initialization import ConfigContext

    ctx = ConfigContext()
    mock_algo = MagicMock()
    ctx.register("Pimple", mock_algo)

    entry: dict[str, Any] = {"type": "spalart_allmaras", "name": "spalart_allmaras"}
    rt = spalart_allmaras.instantiate(case_dir=Path("."), entry=entry)
    rt.run_resolve(ctx)

    assert mock_algo.turbulence_type == "spalart_allmaras"


# --- Cycle 4b: Build ---


def test_build_returns_field_steps() -> None:
    """Build returns InitStep objects for nuTilda, nut, and d."""
    entry: dict[str, Any] = {"type": "spalart_allmaras", "name": "spalart_allmaras"}
    rt = spalart_allmaras.instantiate(case_dir=Path("."), entry=entry)
    steps = rt.run_build()
    assert len(steps) == 3
    step_names = [s.name for s in steps]
    assert "fields.nuTilda" in step_names
    assert "fields.nut" in step_names
    assert "fields.d" in step_names


# --- Cycle 4c: Operation ---


def test_turbulence_correction_operation_registered() -> None:
    """SA has a turbulence_correction operation registered."""
    op_names = [op.name for op in spalart_allmaras._operations]
    assert "turbulence_correction" in op_names


# --- Cycle 5: Module registration in __init__.py ---


def test_registered_via_models_init() -> None:
    """Importing models package registers spalart_allmaras with the plugin system."""
    from neofoam.core.plugin_system import PluginSystem

    # Force import of models __init__ to trigger registration
    import neofoam.solver.incompressibleFluid.models  # noqa: F401

    registry = PluginSystem.get_registered("incompressibleFluidModel")
    assert registry is not None
    plugin_names = [p.__name__ for p in registry.plugin_registry]
    assert "spalart_allmaras" in plugin_names
