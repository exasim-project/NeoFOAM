# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Common test fixtures for 3-stage initialization tests.

This module contains shared test models and configurations used across
multiple test files.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field

from foamadapter.framework.model import Model
from foamadapter.framework.solver import Solver
from foamadapter.framework.initialization import ModelRegistry


# ============================================================================
# Configuration Schemas
# ============================================================================


class TurbulenceConfig(BaseModel):
    """Configuration for turbulence model."""

    model_type: str = Field(default="kEpsilon", description="Type of turbulence model")
    wall_function: bool = Field(default=True, description="Use wall functions")
    coefficients: dict = Field(default_factory=dict, description="Model coefficients")


class TransportConfig(BaseModel):
    """Configuration for transport properties."""

    viscosity: float = Field(default=1e-6, gt=0, description="Kinematic viscosity")
    density: float = Field(default=1000.0, gt=0, description="Fluid density")


class AlgorithmConfig(BaseModel):
    """Configuration for pressure-velocity coupling algorithm."""

    n_correctors: int = Field(default=2, ge=1, description="Number of correctors")
    n_outer_correctors: int = Field(
        default=1, ge=1, description="Number of outer correctors"
    )
    momentum_predictor: bool = Field(default=True, description="Use momentum predictor")


class SolverConfig(BaseModel):
    """Main solver configuration."""

    max_iterations: int = Field(default=100, ge=1, description="Maximum iterations")
    tolerance: float = Field(default=1e-6, gt=0, description="Convergence tolerance")
    time_step: float = Field(default=0.001, gt=0, description="Time step size")


# ============================================================================
# Test Models
# ============================================================================


class TestTurbulenceModel(BaseModel):
    """
    Test turbulence model with 3-stage initialization.

    Demonstrates:
    - Loading coefficients from files (READ_FILES)
    - Connecting to transport model (CONFIGURE)
    - Initializing fields (SETUP)
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = "turbulence"
    config: TurbulenceConfig = Field(default_factory=TurbulenceConfig)

    # State tracking for testing
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False
    transport_ref: Optional[Any] = None  # Reference from CONFIGURE stage

    @Model.read_files
    def load_coefficients(self):
        """READ_FILES: Load turbulence coefficients from file."""
        # Simulate reading from file
        self.config.coefficients = {
            "C_mu": 0.09,
            "C_1": 1.44,
            "C_2": 1.92,
            "sigma_k": 1.0,
            "sigma_epsilon": 1.3,
        }
        self.files_read = True
        return self.config.coefficients

    @Model.configure
    def connect_transport(self, registry: ModelRegistry):
        """CONFIGURE: Connect to transport model for viscosity."""
        # Get reference to transport model
        transport = registry.get("transport")
        if transport:
            self.transport_ref = transport
            self.configured = True
        else:
            raise RuntimeError("Transport model not found in registry")
        return self.configured

    @Model.setup
    def initialize_fields(self, mesh):
        """SETUP: Initialize turbulence fields on mesh."""
        # Create k and epsilon fields
        if self.transport_ref:
            nu = self.transport_ref.config.viscosity
            # Use viscosity for initial turbulence estimates
            # In real code, would create fields on mesh here
            self.setup_complete = True
        return self.setup_complete


class TestTransportModel(BaseModel):
    """
    Test transport properties model.

    Demonstrates:
    - Loading transport properties (READ_FILES)
    - Validating properties (CONFIGURE)
    - Creating coefficient fields (SETUP)
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = "transport"
    config: TransportConfig = Field(default_factory=TransportConfig)

    # State tracking
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False

    @Model.read_files
    def load_properties(self):
        """READ_FILES: Load transport properties from file."""
        # Simulate reading transportProperties
        self.config.viscosity = 1e-6
        self.config.density = 998.0
        self.files_read = True
        return self.config

    @Model.configure
    def validate_properties(self, registry: ModelRegistry):
        """CONFIGURE: Validate transport properties."""
        # Pydantic already validates gt=0, but we can add custom checks
        if self.config.viscosity <= 0:
            raise ValueError("Viscosity must be positive")
        if self.config.density <= 0:
            raise ValueError("Density must be positive")
        self.configured = True
        return self.configured

    @Model.setup
    def create_fields(self, mesh):
        """SETUP: Create transport coefficient fields."""
        # Create nu and rho fields on mesh
        # In real code, would create fields here
        self.setup_complete = True
        return self.setup_complete


class TestAlgorithmModel(BaseModel):
    """
    Test pressure-velocity coupling algorithm.

    Demonstrates:
    - Loading algorithm settings (READ_FILES)
    - Connecting to multiple models (CONFIGURE)
    - Setting up matrix systems (SETUP)
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = "algorithm"
    config: AlgorithmConfig = Field(default_factory=AlgorithmConfig)

    # State tracking
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False
    turbulence_ref: Optional[Any] = None
    transport_ref: Optional[Any] = None

    @Model.read_files
    def load_settings(self):
        """READ_FILES: Load algorithm settings."""
        # Simulate reading from fvSolution
        self.config.n_correctors = 2
        self.config.n_outer_correctors = 1
        self.config.momentum_predictor = True
        self.files_read = True
        return self.config

    @Model.configure
    def connect_models(self, registry: ModelRegistry):
        """CONFIGURE: Connect to turbulence and transport models."""
        self.turbulence_ref = registry.get("turbulence")
        self.transport_ref = registry.get("transport")

        if not self.turbulence_ref:
            raise RuntimeError("Turbulence model not found")
        if not self.transport_ref:
            raise RuntimeError("Transport model not found")

        self.configured = True
        return self.configured

    @Model.setup
    def setup_matrices(self, mesh):
        """SETUP: Set up matrix systems."""
        # In real code, would create matrix structures here
        self.setup_complete = True
        return self.setup_complete


# ============================================================================
# TestSolver
# ============================================================================


class TestSolver(BaseModel):
    """
    Test solver with 3-stage initialization and multiple models.

    Demonstrates the 1 solver with N models architecture pattern.
    """

    model_config = {"arbitrary_types_allowed": True}

    config: SolverConfig = Field(default_factory=SolverConfig)

    # Models (1 solver with N models)
    turbulence: TestTurbulenceModel = Field(default_factory=TestTurbulenceModel)
    transport: TestTransportModel = Field(default_factory=TestTransportModel)
    algorithm: TestAlgorithmModel = Field(default_factory=TestAlgorithmModel)

    # Solver state
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False

    def get_models(self) -> list:
        """Return all models owned by this solver."""
        return [self.turbulence, self.transport, self.algorithm]

    # ---- Solver's own lifecycle methods ----

    @Solver.read_files
    def load_control_dict(self):
        """READ_FILES: Load solver control settings."""
        # Simulate reading controlDict
        self.config.max_iterations = 100
        self.config.tolerance = 1e-6
        self.config.time_step = 0.001
        self.files_read = True
        return self.config

    @Solver.configure
    def validate_config(self, registry: ModelRegistry):
        """CONFIGURE: Validate solver configuration."""
        # Check all models are configured
        for model in self.get_models():
            if not model.configured:
                raise RuntimeError(f"Model {model.name} not configured")
        self.configured = True
        return self.configured

    @Solver.setup
    def create_solver_context(self, mesh):
        """SETUP: Create solver execution context."""
        # In real code, would set up solver runtime structures
        self.setup_complete = True
        return self.setup_complete


# ============================================================================
# Example Models with AdaptableField
# ============================================================================

from foamadapter.framework import AdaptableField


class PressureEquationStandard:
    """Standard pressure equation implementation."""

    def get_operations(self) -> list[str]:
        return ["assemble_momentum", "solve_pressure", "correct_velocity"]


class PressureEquationBuoyant:
    """Buoyancy-modified pressure equation implementation."""

    def get_operations(self) -> list[str]:
        return [
            "assemble_momentum",
            "add_buoyancy",
            "solve_pressure_buoyant",
            "correct_velocity",
        ]


class AdaptivePressureModel(BaseModel):
    """
    Pressure model with AdaptableField for behavior switching.

    Demonstrates how other models can change this model's behavior
    by modifying the use_buoyancy field during CONFIGURE stage.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = "adaptive_pressure"

    # AdaptableField - other models can modify this
    use_buoyancy: bool = AdaptableField(
        default=False, description="Use buoyancy-modified pressure equation"
    )

    # Regular fields
    tolerance: float = Field(default=1e-6, gt=0)
    max_iterations: int = Field(default=50, ge=1)

    # State tracking
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False

    # Implementation dispatch
    _implementations = {False: PressureEquationStandard, True: PressureEquationBuoyant}

    @Model.read_files
    def load_settings(self):
        """READ_FILES: Load pressure solver settings."""
        self.tolerance = 1e-6
        self.max_iterations = 50
        self.files_read = True

    @Model.configure
    def validate(self, registry: ModelRegistry):
        """CONFIGURE: Validate configuration."""
        self.configured = True

    @Model.setup
    def initialize(self, mesh):
        """SETUP: Initialize pressure solver."""
        self.setup_complete = True

    def get_operations(self) -> list[str]:
        """Get operations from selected implementation."""
        impl_class = self._implementations[self.use_buoyancy]
        return impl_class().get_operations()


class TestBuoyancyModel(BaseModel):
    """
    Buoyancy model that adapts pressure model behavior.

    Demonstrates how a model can modify adaptable fields in other models
    during the CONFIGURE stage.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = "test_buoyancy"

    enabled: bool = AdaptableField(default=True)
    beta: float = Field(default=1e-3, description="Thermal expansion coefficient")

    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False

    @Model.read_files
    def load_properties(self):
        """READ_FILES: Load buoyancy properties."""
        self.beta = 1e-3
        self.files_read = True

    @Model.configure
    def adapt_pressure(self, registry: ModelRegistry):
        """CONFIGURE: Tell pressure model to use buoyancy variant."""
        pressure = registry.get("adaptive_pressure")

        if pressure and self.enabled:
            # Modify adaptable field in pressure model
            pressure.use_buoyancy = True

        self.configured = True

    @Model.setup
    def initialize(self, mesh):
        """SETUP: Initialize buoyancy fields."""
        self.setup_complete = True
