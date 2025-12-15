# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for IncompressibleFluid solver 3-stage initialization.

This test suite verifies that the IncompressibleFluid solver properly
implements the 3-stage initialization pattern (LOAD, RESOLVE_DEPENDENCIES, BUILD).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from foamadapter.framework.initialization import (
    SolverInitializer,
    ModelRegistry,
    InitializationStage,
)
from foamadapter.solver.incompressibleFluid import IncompressibleFluid


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_pyfoam():
    """Mock pybFoam module to avoid OpenFOAM dependencies."""
    with patch("foamadapter.solver.incompressibleFluid.pyf") as mock_pyf:
        # Mock dictionary reading - handle controlDict.get[type](key) pattern
        mock_control_dict = MagicMock()

        # Create a mock that handles the [type] subscript and subsequent call
        def mock_get_subscript(self, t):
            def get_value(key):
                values = {
                    "maxDeltaT": 1.0,
                    "adjustTimeStep": True,
                    "maxCo": 0.5,
                }
                if key in values:
                    return values[key]
                raise KeyError(key)

            return get_value

        mock_control_dict.get.__getitem__ = mock_get_subscript

        mock_fv_solution = MagicMock()
        mock_fv_solution.subDict = MagicMock(return_value={})

        mock_pyf.dictionary.read = MagicMock(
            side_effect=lambda path: {
                "system/controlDict": mock_control_dict,
                "system/fvSolution": mock_fv_solution,
            }.get(path)
        )

        # Mock argList, Time, fvMesh with proper mesh object
        mock_pyf.argList = MagicMock(return_value="mock_arglist")
        mock_pyf.Time = MagicMock(return_value="mock_runtime")

        mock_mesh = MagicMock()
        mock_mesh.setFluxRequired = MagicMock()
        mock_pyf.fvMesh = MagicMock(return_value=mock_mesh)

        # Mock field reading
        mock_pyf.createPhi = MagicMock(return_value="mock_phi")
        mock_pyf.setRefCell = MagicMock(return_value=(0, 0.0))
        mock_pyf.Word = MagicMock(return_value="mock_word")

        yield mock_pyf


@pytest.fixture
def mock_field_classes():
    """Mock field classes."""
    with (
        patch(
            "foamadapter.solver.incompressibleFluid.volScalarField"
        ) as mock_vol_scalar,
        patch(
            "foamadapter.solver.incompressibleFluid.volVectorField"
        ) as mock_vol_vector,
    ):
        mock_vol_scalar.read_field = MagicMock(return_value="mock_p_field")
        mock_vol_vector.read_field = MagicMock(return_value="mock_U_field")

        yield mock_vol_scalar, mock_vol_vector


@pytest.fixture
def mock_turbulence_models():
    """Mock turbulence model classes."""
    with (
        patch(
            "foamadapter.solver.incompressibleFluid.singlePhaseTransportModel"
        ) as mock_transport,
        patch(
            "foamadapter.solver.incompressibleFluid.incompressibleTurbulenceModel"
        ) as mock_turbulence,
    ):
        mock_transport.return_value = "mock_transport_model"
        mock_turbulence.New = MagicMock(return_value="mock_turbulence_model")

        yield mock_transport, mock_turbulence


@pytest.fixture
def mock_builder():
    """Create a mock ContextBuilder for tests."""
    builder = Mock()
    builder.add_field = Mock()
    builder.add_model = Mock()
    builder.set_mesh = Mock()
    builder.set_runtime = Mock()
    builder.build = Mock(return_value="mock_context")
    return builder


@pytest.fixture
def solver_basic():
    """Create a basic IncompressibleFluid solver instance."""
    return IncompressibleFluid(argv=["test"], algorithm="PIMPLE")


# ============================================================================
# Tests for Solver Creation
# ============================================================================


def test_solver_creation():
    """Test that solver can be created with default parameters."""
    solver = IncompressibleFluid()

    assert solver.name == "IncompressibleFluid"
    assert solver.algorithm == "PIMPLE"
    assert solver.files_read is False
    assert solver.configured is False
    assert solver.setup_complete is False


def test_solver_creation_with_custom_params():
    """Test solver creation with custom parameters."""
    solver = IncompressibleFluid(
        argv=["test_app", "-case", "/path/to/case"],
        algorithm="PIMPLE",
        maxDeltaT=0.1,
    )

    assert solver.argv == ["test_app", "-case", "/path/to/case"]
    assert solver.algorithm == "PIMPLE"
    assert solver.maxDeltaT == 0.1


def test_get_models_initially_empty():
    """Test that get_models returns empty list before initialization."""
    solver = IncompressibleFluid()
    models = solver.get_models()

    assert isinstance(models, list)
    assert len(models) == 0


# ============================================================================
# Tests for LOAD Stage
# ============================================================================


def test_load_stage_execution(solver_basic, mock_pyfoam):
    """Test that LOAD stage executes correctly."""
    solver_basic.load_control_dict()

    assert solver_basic.files_read is True
    # maxDeltaT should be updated from mock
    assert solver_basic.maxDeltaT == 1.0


def test_load_handles_missing_maxDeltaT(solver_basic):
    """Test that missing maxDeltaT doesn't break initialization."""
    with patch("foamadapter.solver.incompressibleFluid.pyf") as mock_pyf:
        # Mock KeyError for maxDeltaT
        mock_control_dict = MagicMock()

        def mock_get_subscript_with_error(self, t):
            def get_value(key):
                raise KeyError(key)

            return get_value

        mock_control_dict.get.__getitem__ = mock_get_subscript_with_error
        mock_pyf.dictionary.read.return_value = mock_control_dict

        original_max_delta_t = solver_basic.maxDeltaT
        solver_basic.load_control_dict()

        assert solver_basic.files_read is True
        assert solver_basic.maxDeltaT == original_max_delta_t  # Should keep default


def test_load_decorator_marked():
    """Test that load_control_dict is marked with LOAD stage."""
    solver = IncompressibleFluid()
    method = solver.load_control_dict

    assert hasattr(method, "_init_stage")
    assert method._init_stage == InitializationStage.LOAD


# ============================================================================
# Tests for RESOLVE_DEPENDENCIES Stage
# ============================================================================


def test_resolve_dependencies_stage_execution(solver_basic):
    """Test that RESOLVE_DEPENDENCIES stage executes correctly."""
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    assert solver_basic.configured is True
    # Components are created as wrappers in RESOLVE_DEPENDENCIES
    assert solver_basic._transport is not None
    assert solver_basic._turbulence is not None
    # Algorithm is created later in BUILD after pRefCell/pRefValue are known
    assert solver_basic._pressure_velocity is None
    assert registry.contains("transport")
    assert registry.contains("turbulence")


def test_resolve_dependencies_registers_algorithm(solver_basic):
    """Test that components are registered in ModelRegistry."""
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    # New architecture: components are registered, not algorithm
    transport = registry.get("transport")
    turbulence = registry.get("turbulence")
    assert transport is not None
    assert turbulence is not None
    assert transport is solver_basic._transport
    assert turbulence is solver_basic._turbulence


def test_resolve_dependencies_validates_algorithm():
    """Test that invalid algorithm raises ValueError."""
    # Note: Pydantic validation happens at construction time
    # We test the configure stage validation
    solver = IncompressibleFluid(algorithm="PIMPLE")
    solver.algorithm = "INVALID"  # Bypass Pydantic for testing

    registry = ModelRegistry()
    with pytest.raises(ValueError, match="Unknown algorithm"):
        solver.configure_solver(registry)


def test_resolve_dependencies_decorator_marked():
    """Test that configure_solver is marked with RESOLVE_DEPENDENCIES stage."""
    solver = IncompressibleFluid()
    method = solver.configure_solver

    assert hasattr(method, "_init_stage")
    assert method._init_stage == InitializationStage.RESOLVE_DEPENDENCIES


# ============================================================================
# Tests for BUILD Stage
# ============================================================================


def test_build_stage_execution(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models, mock_builder
):
    """Test that BUILD stage executes correctly."""
    # Must call configure_solver first
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    solver_basic.setup_runtime(mesh=None, builder=mock_builder)

    assert solver_basic.setup_complete is True
    # Transport and turbulence wrappers are created in RESOLVE_DEPENDENCIES
    assert solver_basic._transport is not None
    assert solver_basic._turbulence is not None
    # Algorithm is created in BUILD
    assert solver_basic._pressure_velocity is not None


def test_build_creates_mesh_and_runtime(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models, mock_builder
):
    """Test that BUILD stage creates mesh and runtime objects."""
    # Must call configure_solver first
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    solver_basic.setup_runtime(mesh=None, builder=mock_builder)

    mock_pyfoam.argList.assert_called_once_with(solver_basic.argv)
    mock_pyfoam.Time.assert_called_once()
    mock_pyfoam.fvMesh.assert_called_once()


def test_build_reads_fields(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models, mock_builder
):
    """Test that BUILD stage reads pressure and velocity fields."""
    mock_vol_scalar, mock_vol_vector = mock_field_classes

    # Must call configure_solver first
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    solver_basic.setup_runtime(mesh=None, builder=mock_builder)

    # Should read p and U fields
    mock_vol_scalar.read_field.assert_called()
    mock_vol_vector.read_field.assert_called()


def test_build_creates_turbulence_models(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models, mock_builder
):
    """Test that BUILD stage creates transport and turbulence models."""
    mock_transport, mock_turbulence = mock_turbulence_models

    # Must call configure_solver first
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    solver_basic.setup_runtime(mesh=None, builder=mock_builder)

    mock_transport.assert_called()
    mock_turbulence.New.assert_called()
    # _transport and _turbulence are wrappers created in RESOLVE_DEPENDENCIES
    assert solver_basic._transport is not None
    assert solver_basic._turbulence is not None


def test_build_updates_algorithm_reference_cell(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models, mock_builder
):
    """Test that BUILD stage creates algorithm with reference cell/value."""
    # Must call configure_solver first
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    solver_basic.setup_runtime(mesh=None, builder=mock_builder)

    # Algorithm is created in BUILD with pRefCell and pRefValue
    assert solver_basic._pressure_velocity is not None
    assert solver_basic._pressure_velocity.pRefCell == 0
    assert solver_basic._pressure_velocity.pRefValue == 0.0


def test_build_decorator_marked():
    """Test that setup_runtime is marked with BUILD stage."""
    solver = IncompressibleFluid()
    method = solver.setup_runtime

    assert hasattr(method, "_init_stage")
    assert method._init_stage == InitializationStage.BUILD


# ============================================================================
# Tests for Full Initialization Flow
# ============================================================================


def test_full_initialization_with_initializer(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test complete 3-stage initialization using SolverInitializer."""
    initializer = SolverInitializer(solver_basic)
    result = initializer.initialize(mesh=None)

    # All stages should be complete
    assert solver_basic.files_read
    assert solver_basic.configured
    assert solver_basic.setup_complete

    # Should return a Context now (not the solver)
    from foamadapter.framework.context import Context

    assert isinstance(result, Context)
    assert result.mesh is not None
    assert result.runTime is not None


def test_initialization_order(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test that initialization stages execute in correct order."""
    call_order = []

    # Use side_effect to track calls
    original_read = IncompressibleFluid.load_control_dict
    original_configure = IncompressibleFluid.configure_solver
    original_setup = IncompressibleFluid.setup_runtime

    def tracked_read(self):
        call_order.append("LOAD")
        return original_read(self)

    def tracked_configure(self, registry):
        call_order.append("RESOLVE_DEPENDENCIES")
        return original_configure(self, registry)

    def tracked_setup(self, mesh, builder):
        call_order.append("BUILD")
        return original_setup(self, mesh, builder)

    # Preserve decorators by copying _init_stage attribute
    tracked_read._init_stage = InitializationStage.LOAD
    tracked_configure._init_stage = InitializationStage.RESOLVE_DEPENDENCIES
    tracked_setup._init_stage = InitializationStage.BUILD

    with patch.object(IncompressibleFluid, "load_control_dict", tracked_read):
        with patch.object(IncompressibleFluid, "configure_solver", tracked_configure):
            with patch.object(IncompressibleFluid, "setup_runtime", tracked_setup):
                initializer = SolverInitializer(solver_basic)
                initializer.initialize(mesh=None)

    assert call_order == ["LOAD", "RESOLVE_DEPENDENCIES", "BUILD"]


def test_get_models_after_resolve_dependencies(solver_basic, mock_pyfoam):
    """Test that get_models returns empty list after RESOLVE_DEPENDENCIES stage."""
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    # get_models() returns optional physics models, not core components
    models = solver_basic.get_models()
    assert len(models) == 0


# ============================================================================
# Tests for Context from Initialization
# ============================================================================


def test_context_from_initialization(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test that initialization returns a context with all required fields."""
    # Initialize the solver
    initializer = SolverInitializer(solver_basic)
    ctx = initializer.initialize(mesh=None)

    # Verify context has all required components
    assert ctx is not None
    assert ctx.mesh is not None
    assert ctx.runTime is not None
    assert "p" in ctx.fields
    assert "U" in ctx.fields
    assert "phi" in ctx.fields
    assert "laminarTransport" in ctx.fields
    assert "turbulence" in ctx.fields
    assert "pimple" in ctx.fields


# ============================================================================
# Tests for Algorithm Creation via PluginSystem
# ============================================================================


def test_algorithm_created_in_build():
    """Test that algorithm is created during BUILD stage."""
    solver = IncompressibleFluid(algorithm="PIMPLE", pRefCell=5, pRefValue=100.0)

    # Algorithm should not exist before initialization
    assert solver._pressure_velocity is None

    # After proper initialization, algorithm should be created
    # This is tested in the full lifecycle tests


# ============================================================================
# Tests for Model Registry Integration
# ============================================================================


def test_registry_contains_algorithm_after_resolve_dependencies(solver_basic):
    """Test that ModelRegistry contains components after RESOLVE_DEPENDENCIES."""
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    # New architecture: components are registered
    assert registry.contains("transport")
    assert registry.contains("turbulence")
    transport = registry.get("transport")
    turbulence = registry.get("turbulence")
    assert transport is solver_basic._transport
    assert turbulence is solver_basic._turbulence


def test_registry_empty_before_resolve_dependencies(solver_basic):
    """Test that ModelRegistry is empty before RESOLVE_DEPENDENCIES."""
    registry = ModelRegistry()

    assert not registry.contains("transport")
    assert not registry.contains("turbulence")
    assert registry.get("transport") is None
    assert registry.get("turbulence") is None


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_solver_lifecycle(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test complete solver lifecycle from creation to initialization."""
    # 1. Solver creation
    assert not solver_basic.files_read
    assert not solver_basic.configured
    assert not solver_basic.setup_complete

    # 2. Initialize with SolverInitializer
    initializer = SolverInitializer(solver_basic)
    initializer.initialize(mesh=None)

    # 3. Verify all stages completed
    assert solver_basic.files_read
    assert solver_basic.configured
    assert solver_basic.setup_complete

    # 4. Verify core components are initialized
    assert solver_basic._transport is not None
    assert solver_basic._turbulence is not None
    assert solver_basic._pressure_velocity is not None

    # 5. Verify registry has components
    assert initializer.registry.contains("transport")
    assert initializer.registry.contains("turbulence")


def test_multiple_initializations_idempotent(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test that multiple initializations don't break the solver."""
    initializer = SolverInitializer(solver_basic)

    # First initialization
    initializer.initialize(mesh=None)
    first_algorithm = solver_basic._pressure_velocity

    # Second initialization (should work without errors)
    initializer.initialize(mesh=None)
    second_algorithm = solver_basic._pressure_velocity

    # Both should have created algorithms (may be different instances)
    assert first_algorithm is not None
    assert second_algorithm is not None
