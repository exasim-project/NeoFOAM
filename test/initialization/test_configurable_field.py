# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Tests for Configurable field feature.

Tests the ability of models to expose behavior switches that other models
can modify during RESOLVE_DEPENDENCIES stage, with automatic dispatch to different
implementations.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - framework refactored (SolverInitializer removed)"
)

# Prevent code execution since test is skipped
import sys

if True:  # Always skip
    pytest.skip("Outdated - framework refactored", allow_module_level=True)

# from pydantic import BaseModel, Field
#
# from foamadapter.framework.initialization import (
#     Configurable,
#     ConfigContext,
#     SolverInitializer,
# )
# from foamadapter.framework.model import Model


# ============================================================================
# Test Implementations
# ============================================================================


class StandardPressureImpl:
    """Standard pressure equation without buoyancy."""

    def get_operations(self) -> list[str]:
        return ["assemble_momentum", "solve_pressure", "correct_velocity"]


class BuoyantPressureImpl:
    """Pressure equation with buoyancy terms."""

    def get_operations(self) -> list[str]:
        return [
            "assemble_momentum",
            "add_buoyancy_source",
            "solve_pressure_buoyant",
            "correct_velocity",
        ]


class SIMPLEAlgorithm:
    """SIMPLE algorithm implementation."""

    def get_operations(self) -> list[str]:
        return ["momentum", "pressure_simple", "correct"]


class PISOAlgorithm:
    """PISO algorithm implementation."""

    def get_operations(self) -> list[str]:
        return ["momentum", "pressure_piso", "corrector_loop"]


class PIMPLEAlgorithm:
    """PIMPLE algorithm implementation."""

    def get_operations(self) -> list[str]:
        return ["outer_loop", "momentum", "pressure_pimple", "corrector_loop"]


# ============================================================================
# Test Models with Configurable
# ============================================================================


class PressureAlgorithmSimple(BaseModel):
    """Pressure algorithm with single configurable field (boolean switch)."""

    model_config = {"arbitrary_types_allowed": True}

    name: str = "pressure_algorithm"

    # Configurable field - can be modified by other models
    use_buoyancy: Configurable[bool] = False

    # Regular fields - not configurable
    tolerance: float = Field(default=1e-6, gt=0)
    max_iterations: int = Field(default=50, ge=1)

    # Implementation registry
    _implementations = {False: StandardPressureImpl, True: BuoyantPressureImpl}

    def get_operations(self) -> list[str]:
        """Dispatch to implementation based on configurable field."""
        impl_class = self._implementations[self.use_buoyancy]
        return impl_class().get_operations()


class PressureAlgorithmMulti(BaseModel):
    """Pressure algorithm with multiple configurable fields."""

    model_config = {"arbitrary_types_allowed": True}

    name: str = "pressure_velocity"

    # Configurable fields
    algorithm: Configurable[str] = "SIMPLE"
    use_buoyancy: Configurable[bool] = False

    # Regular field
    n_correctors: int = Field(default=2, ge=1)

    # Implementation registry (tuple key)
    _implementations = {
        ("SIMPLE", False): SIMPLEAlgorithm,
        ("SIMPLE", True): SIMPLEAlgorithm,  # Could be different variant
        ("PISO", False): PISOAlgorithm,
        ("PISO", True): PISOAlgorithm,
        ("PIMPLE", False): PIMPLEAlgorithm,
        ("PIMPLE", True): PIMPLEAlgorithm,
    }

    def get_operations(self) -> list[str]:
        """Dispatch based on (algorithm, use_buoyancy) tuple."""
        key = (self.algorithm, self.use_buoyancy)
        impl_class = self._implementations.get(key)
        if not impl_class:
            raise ValueError(f"No implementation for {key}")
        return impl_class().get_operations()


class BuoyancyModel(BaseModel):
    """Buoyancy model that modifies pressure algorithm."""

    model_config = {"arbitrary_types_allowed": True}

    name: str = "buoyancy"

    enabled: Configurable[bool] = True
    beta: float = Field(default=1e-3, description="Thermal expansion")

    configured: bool = False

    @Model.resolve_dependencies
    def resolve_dependencies(self, config: ConfigContext):
        """Tell pressure algorithm to use buoyancy variant."""
        pressure = config.get("pressure_algorithm")

        if pressure and self.enabled:
            # Modify configurable field in another model
            pressure.use_buoyancy = True

        self.configured = True


class HeatSource(BaseModel):
    """Heat source model - multiple instances possible."""

    model_config = {"arbitrary_types_allowed": True}

    name: str  # e.g., "heat_source_1", "heat_source_2"

    # Configurable field
    enabled: Configurable[bool] = True

    # Regular fields
    power: float = Field(default=1000.0, gt=0)
    location: tuple = Field(default=(0, 0, 0))

    def get_operations(self) -> list[str]:
        """Return operations if enabled."""
        if self.enabled:
            return [f"add_heat_source_{self.name}"]
        return []


# ============================================================================
# Tests
# ============================================================================


def test_configurable_field_type_annotation():
    """Test that Configurable creates proper type annotation."""
    from foamadapter.framework.initialization.configurable import is_configurable_field

    model = PressureAlgorithmSimple()

    # Check that configurable field is detected
    field_info = model.__class__.model_fields["use_buoyancy"]
    assert is_configurable_field(field_info)

    # Check that regular fields are not detected as configurable
    tolerance_info = model.__class__.model_fields["tolerance"]
    assert not is_configurable_field(tolerance_info)


def test_single_configurable_field_dispatch():
    """Test dispatch with single boolean configurable field."""
    model = PressureAlgorithmSimple()

    # Default behavior (no buoyancy)
    ops = model.get_operations()
    assert "solve_pressure" in ops
    assert "add_buoyancy_source" not in ops

    # Switch to buoyancy variant
    model.use_buoyancy = True
    ops = model.get_operations()
    assert "solve_pressure_buoyant" in ops
    assert "add_buoyancy_source" in ops


def test_multiple_configurable_fields_dispatch():
    """Test dispatch with multiple configurable fields (tuple key)."""
    model = PressureAlgorithmMulti()

    # SIMPLE without buoyancy
    assert model.algorithm == "SIMPLE"
    assert model.use_buoyancy is False
    ops = model.get_operations()
    assert "pressure_simple" in ops

    # Switch to PISO
    model.algorithm = "PISO"
    ops = model.get_operations()
    assert "pressure_piso" in ops
    assert "corrector_loop" in ops

    # Switch to PIMPLE with buoyancy
    model.algorithm = "PIMPLE"
    model.use_buoyancy = True
    ops = model.get_operations()
    assert "pressure_pimple" in ops
    assert "outer_loop" in ops


def test_cross_model_configuration():
    """Test that one model can configure another via configurable fields."""
    pressure = PressureAlgorithmSimple()
    buoyancy = BuoyancyModel()

    class TestSolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        pressure: PressureAlgorithmSimple
        buoyancy: BuoyancyModel

        def get_models(self):
            return [self.pressure, self.buoyancy]

    solver = TestSolver(pressure=pressure, buoyancy=buoyancy)
    initializer = SolverInitializer(solver)

    # Before RESOLVE_DEPENDENCIES: pressure uses standard implementation
    assert pressure.use_buoyancy is False
    ops_before = pressure.get_operations()
    assert "solve_pressure" in ops_before
    assert "add_buoyancy_source" not in ops_before

    # Run initialization (buoyancy.resolve_dependencies modifies pressure.use_buoyancy)
    initializer.initialize()

    # After RESOLVE_DEPENDENCIES: pressure uses buoyancy implementation
    assert pressure.use_buoyancy is True
    ops_after = pressure.get_operations()
    assert "solve_pressure_buoyant" in ops_after
    assert "add_buoyancy_source" in ops_after
    assert buoyancy.configured is True


def test_config_context_get_configurable_fields():
    """Test ConfigContext.get_configurable_fields() method."""
    pressure = PressureAlgorithmSimple(use_buoyancy=False)
    buoyancy = BuoyancyModel(enabled=True)

    config = ConfigContext()
    config.register("pressure_algorithm", pressure)
    config.register("buoyancy", buoyancy)

    # Get configurable fields from pressure algorithm
    configurable = config.get_configurable_fields("pressure_algorithm")
    assert "use_buoyancy" in configurable
    assert configurable["use_buoyancy"] is False
    assert "tolerance" not in configurable  # Regular field, not configurable

    # Get configurable fields from buoyancy
    configurable_buoy = config.get_configurable_fields("buoyancy")
    assert "enabled" in configurable_buoy
    assert configurable_buoy["enabled"] is True
    assert "beta" not in configurable_buoy  # Regular field


def test_config_context_get_by_type():
    """Test ConfigContext.get_by_type() for multiple instances."""
    source1 = HeatSource(name="heat_source_1", power=1000.0)
    source2 = HeatSource(name="heat_source_2", power=500.0)
    source3 = HeatSource(name="heat_source_3", power=2000.0)

    config = ConfigContext()
    config.register("heat_source_1", source1)
    config.register("heat_source_2", source2)
    config.register("heat_source_3", source3)

    # Get all heat sources by type
    heat_sources = config.get_by_type(HeatSource)
    assert len(heat_sources) == 3
    assert source1 in heat_sources
    assert source2 in heat_sources
    assert source3 in heat_sources


def test_config_context_get_by_prefix():
    """Test ConfigContext.get_by_prefix() for multiple instances."""
    source1 = HeatSource(name="heat_source_1", power=1000.0)
    source2 = HeatSource(name="heat_source_2", power=500.0)
    other = PressureAlgorithmSimple(name="pressure")

    config = ConfigContext()
    config.register("heat_source_1", source1)
    config.register("heat_source_2", source2)
    config.register("pressure", other)

    # Get by prefix
    sources = config.get_by_prefix("heat_source_")
    assert len(sources) == 2
    assert "heat_source_1" in sources
    assert "heat_source_2" in sources
    assert "pressure" not in sources


def test_multiple_instances_configuration():
    """Test configuring multiple instances of same model type."""
    source1 = HeatSource(name="heat_source_1", power=1000.0, location=(0, 0, 0))
    source2 = HeatSource(name="heat_source_2", power=500.0, location=(5, 0, 0))
    source3 = HeatSource(name="heat_source_3", power=2000.0, location=(10, 0, 0))

    class ControlModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "control"

        @Model.resolve_dependencies
        def resolve_dependencies(self, config: ConfigContext):
            # Disable heat sources outside certain region
            sources = config.get_by_type(HeatSource)
            for source in sources:
                if source.location[0] > 7:
                    source.enabled = False

    class TestSolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        control: ControlModel
        sources: list[HeatSource]

        def get_models(self):
            return [self.control] + self.sources

    control = ControlModel()
    solver = TestSolver(control=control, sources=[source1, source2, source3])
    initializer = SolverInitializer(solver)

    # Before RESOLVE_DEPENDENCIES: all enabled
    assert source1.enabled is True
    assert source2.enabled is True
    assert source3.enabled is True

    # Run initialization
    initializer.initialize()

    # After RESOLVE_DEPENDENCIES: source3 disabled (location[0] > 7)
    assert source1.enabled is True
    assert source2.enabled is True
    assert source3.enabled is False

    # Check operations
    ops1 = source1.get_operations()
    ops3 = source3.get_operations()
    assert len(ops1) == 1  # Enabled
    assert len(ops3) == 0  # Disabled


def test_configurable_field_validation():
    """Test that Pydantic validation works with Configurable."""
    model = PressureAlgorithmMulti()

    # Valid values
    model.algorithm = "PISO"
    assert model.algorithm == "PISO"

    # Pydantic validation happens during model creation, not assignment
    # Test that creating a model with invalid values fails
    with pytest.raises(Exception):
        PressureAlgorithmMulti(n_correctors=0)  # Should fail (ge=1)


def test_configurable_field_with_complex_dispatch():
    """Test complex dispatch logic with multiple configurable fields."""
    model = PressureAlgorithmMulti()

    # Test all combinations
    test_cases = [
        ("SIMPLE", False, "pressure_simple"),
        ("PISO", False, "pressure_piso"),
        ("PIMPLE", False, "pressure_pimple"),
    ]

    for algo, buoyancy, expected_op in test_cases:
        model.algorithm = algo
        model.use_buoyancy = buoyancy
        ops = model.get_operations()
        assert expected_op in ops, (
            f"Expected {expected_op} in operations for {algo}, buoyancy={buoyancy}"
        )


def test_configurable_field_independence():
    """Test that models without configurable fields still work."""

    class SimpleModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "simple"
        value: float = Field(default=1.0)

    model = SimpleModel()
    config = ConfigContext()
    config.register("simple", model)

    # Should return empty dict (no configurable fields)
    configurable = config.get_configurable_fields("simple")
    assert len(configurable) == 0
