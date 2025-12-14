# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for IncompressibleFluid solver 3-stage initialization.

This test suite verifies that the IncompressibleFluid solver properly
implements the 3-stage initialization pattern (READ_FILES, CONFIGURE, SETUP).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from foamadapter.framework.initialization import SolverInitializer, ModelRegistry
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
# Tests for READ_FILES Stage
# ============================================================================


def test_read_files_stage_execution(solver_basic, mock_pyfoam):
    """Test that READ_FILES stage executes correctly."""
    solver_basic.load_control_dict()

    assert solver_basic.files_read is True
    # maxDeltaT should be updated from mock
    assert solver_basic.maxDeltaT == 1.0


def test_read_files_handles_missing_maxDeltaT(solver_basic):
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


def test_read_files_decorator_marked():
    """Test that load_control_dict is marked with READ_FILES stage."""
    solver = IncompressibleFluid()
    method = solver.load_control_dict

    assert hasattr(method, "_init_stage")
    assert method._init_stage == "READ_FILES"


# ============================================================================
# Tests for CONFIGURE Stage
# ============================================================================


def test_configure_stage_execution(solver_basic):
    """Test that CONFIGURE stage executes correctly."""
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    assert solver_basic.configured is True
    assert solver_basic.algorithm_model is not None
    assert registry.contains("algorithm")


def test_configure_registers_algorithm(solver_basic):
    """Test that algorithm is registered in ModelRegistry."""
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    algorithm = registry.get("algorithm")
    assert algorithm is not None
    assert algorithm is solver_basic.algorithm_model


def test_configure_validates_algorithm():
    """Test that invalid algorithm raises ValueError."""
    # Note: Pydantic validation happens at construction time
    # We test the configure stage validation
    solver = IncompressibleFluid(algorithm="PIMPLE")
    solver.algorithm = "INVALID"  # Bypass Pydantic for testing

    registry = ModelRegistry()
    with pytest.raises(ValueError, match="not yet implemented"):
        solver.configure_solver(registry)


def test_configure_decorator_marked():
    """Test that configure_solver is marked with CONFIGURE stage."""
    solver = IncompressibleFluid()
    method = solver.configure_solver

    assert hasattr(method, "_init_stage")
    assert method._init_stage == "CONFIGURE"


# ============================================================================
# Tests for SETUP Stage
# ============================================================================


def test_setup_stage_execution(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test that SETUP stage executes correctly."""
    solver_basic.setup_runtime(mesh=None)

    assert solver_basic.setup_complete is True
    assert solver_basic.mesh is not None
    assert solver_basic.runTime is not None
    assert solver_basic.transport_model is not None
    assert solver_basic.turbulence_model is not None


def test_setup_creates_mesh_and_runtime(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test that SETUP stage creates mesh and runtime objects."""
    solver_basic.setup_runtime(mesh=None)

    mock_pyfoam.argList.assert_called_once_with(solver_basic.argv)
    mock_pyfoam.Time.assert_called_once()
    mock_pyfoam.fvMesh.assert_called_once()


def test_setup_reads_fields(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test that SETUP stage reads pressure and velocity fields."""
    mock_vol_scalar, mock_vol_vector = mock_field_classes

    solver_basic.setup_runtime(mesh=None)

    # Should read p and U fields
    mock_vol_scalar.read_field.assert_called()
    mock_vol_vector.read_field.assert_called()


def test_setup_creates_turbulence_models(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test that SETUP stage creates transport and turbulence models."""
    mock_transport, mock_turbulence = mock_turbulence_models

    solver_basic.setup_runtime(mesh=None)

    mock_transport.assert_called_once()
    mock_turbulence.New.assert_called_once()
    assert solver_basic.transport_model == "mock_transport_model"
    assert solver_basic.turbulence_model == "mock_turbulence_model"


def test_setup_updates_algorithm_reference_cell(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test that SETUP stage updates algorithm with reference cell/value."""
    solver_basic.algorithm_model = Mock()
    solver_basic.setup_runtime(mesh=None)

    # Should update algorithm model with pRefCell and pRefValue
    assert solver_basic.algorithm_model.pRefCell == 0
    assert solver_basic.algorithm_model.pRefValue == 0.0


def test_setup_decorator_marked():
    """Test that setup_runtime is marked with SETUP stage."""
    solver = IncompressibleFluid()
    method = solver.setup_runtime

    assert hasattr(method, "_init_stage")
    assert method._init_stage == "SETUP"


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

    # Should return the solver
    assert result is solver_basic


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
        call_order.append("READ_FILES")
        return original_read(self)

    def tracked_configure(self, registry):
        call_order.append("CONFIGURE")
        return original_configure(self, registry)

    def tracked_setup(self, mesh):
        call_order.append("SETUP")
        return original_setup(self, mesh)

    # Preserve decorators by copying _init_stage attribute
    tracked_read._init_stage = "READ_FILES"
    tracked_configure._init_stage = "CONFIGURE"
    tracked_setup._init_stage = "SETUP"

    with patch.object(IncompressibleFluid, "load_control_dict", tracked_read):
        with patch.object(IncompressibleFluid, "configure_solver", tracked_configure):
            with patch.object(IncompressibleFluid, "setup_runtime", tracked_setup):
                initializer = SolverInitializer(solver_basic)
                initializer.initialize(mesh=None)

    assert call_order == ["READ_FILES", "CONFIGURE", "SETUP"]


def test_get_models_after_configure(solver_basic, mock_pyfoam):
    """Test that get_models returns algorithm after CONFIGURE stage."""
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    models = solver_basic.get_models()
    assert len(models) == 1
    assert solver_basic.algorithm_model in models


# ============================================================================
# Tests for create_context Integration
# ============================================================================


def test_create_context_after_initialization(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test that create_context works after 3-stage initialization."""
    # Initialize the solver
    initializer = SolverInitializer(solver_basic)
    initializer.initialize(mesh=None)

    # Mock pimpleControl
    with patch(
        "foamadapter.solver.incompressibleFluid.pyf.pimpleControl"
    ) as mock_pimple:
        mock_pimple.return_value = "mock_pimple_control"

        # Should be able to create context
        ctx = solver_basic.create_context()

        assert ctx is not None
        assert ctx.mesh is not None
        assert ctx.runTime is not None
        assert "p" in ctx.fields
        assert "U" in ctx.fields
        assert "phi" in ctx.fields
        assert "laminarTransport" in ctx.fields
        assert "turbulence" in ctx.fields


def test_create_context_before_initialization_fails(solver_basic):
    """Test that create_context fails before initialization."""
    with pytest.raises(RuntimeError, match="not properly initialized"):
        solver_basic.create_context()


# ============================================================================
# Tests for Algorithm Creation
# ============================================================================


def test_create_algorithm_pimple():
    """Test that PIMPLE algorithm is created correctly."""
    solver = IncompressibleFluid(algorithm="PIMPLE", pRefCell=5, pRefValue=100.0)
    algorithm = solver._create_algorithm()

    assert algorithm is not None
    assert algorithm.pRefCell == 5
    assert algorithm.pRefValue == 100.0


def test_create_algorithm_unsupported():
    """Test that unsupported algorithms raise ValueError."""
    solver = IncompressibleFluid(algorithm="PIMPLE")
    solver.algorithm = "SIMPLE"  # Not yet implemented

    with pytest.raises(ValueError, match="not yet implemented"):
        solver._create_algorithm()


# ============================================================================
# Tests for Model Registry Integration
# ============================================================================


def test_registry_contains_algorithm_after_configure(solver_basic):
    """Test that ModelRegistry contains algorithm after CONFIGURE."""
    registry = ModelRegistry()
    solver_basic.configure_solver(registry)

    assert registry.contains("algorithm")
    algorithm = registry.get("algorithm")
    assert algorithm is solver_basic.algorithm_model


def test_registry_empty_before_configure(solver_basic):
    """Test that ModelRegistry is empty before CONFIGURE."""
    registry = ModelRegistry()

    assert not registry.contains("algorithm")
    assert registry.get("algorithm") is None


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

    # 4. Verify models are available
    models = solver_basic.get_models()
    assert len(models) == 1
    assert solver_basic.algorithm_model in models

    # 5. Verify registry has algorithm
    assert initializer.registry.contains("algorithm")


def test_multiple_initializations_idempotent(
    solver_basic, mock_pyfoam, mock_field_classes, mock_turbulence_models
):
    """Test that multiple initializations don't break the solver."""
    initializer = SolverInitializer(solver_basic)

    # First initialization
    initializer.initialize(mesh=None)
    first_algorithm = solver_basic.algorithm_model

    # Second initialization (should work without errors)
    initializer.initialize(mesh=None)
    second_algorithm = solver_basic.algorithm_model

    # Both should have created algorithms (may be different instances)
    assert first_algorithm is not None
    assert second_algorithm is not None
