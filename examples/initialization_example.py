# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Example: Using 3-Stage Initialization

This example demonstrates how to use the 3-stage initialization system
with a simple incompressible flow solver.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field

from foamadapter.framework.model import Model
from foamadapter.framework.solver import Solver
from foamadapter.framework.initialization import (
    ModelRegistry,
    SolverInitializer,
)


# ============================================================================
# Configuration Schemas
# ============================================================================

class FluidConfig(BaseModel):
    """Fluid properties configuration."""
    viscosity: float = Field(default=1e-6, gt=0)
    density: float = Field(default=1000.0, gt=0)


class SolverConfig(BaseModel):
    """Solver control configuration."""
    max_iterations: int = Field(default=50, ge=1)
    tolerance: float = Field(default=1e-5, gt=0)


# ============================================================================
# Models
# ============================================================================

class FluidPropertiesModel(BaseModel):
    """Model for fluid transport properties."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    name: str = "fluid"
    config: FluidConfig = Field(default_factory=FluidConfig)
    
    @Model.load
    def load_properties(self):
        """Load properties from transportProperties file."""
        print(f"[{self.name}] LOAD: Loading fluid properties...")
        # In real code: read from file
        self.config.viscosity = 1e-6
        self.config.density = 998.0
    
    @Model.resolve_dependencies
    def validate_properties(self, registry: ModelRegistry):
        """Validate that properties are physical."""
        print(f"[{self.name}] RESOLVE_DEPENDENCIES: Validating properties...")
        if self.config.viscosity <= 0:
            raise ValueError("Viscosity must be positive")
    
    @Model.build
    def create_fields(self, mesh):
        """Create viscosity and density fields."""
        print(f"[{self.name}] BUILD: Creating nu and rho fields...")
        # In real code: create fields on mesh


class MomentumModel(BaseModel):
    """Model for momentum equation."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    name: str = "momentum"
    fluid_ref: Optional[Any] = None
    
    @Model.load
    def load_schemes(self):
        """Load discretization schemes."""
        print(f"[{self.name}] LOAD: Loading schemes from fvSchemes...")
    
    @Model.resolve_dependencies
    def connect_fluid(self, registry: ModelRegistry):
        """Connect to fluid properties model."""
        print(f"[{self.name}] RESOLVE_DEPENDENCIES: Connecting to fluid properties...")
        self.fluid_ref = registry.get("fluid")
        if not self.fluid_ref:
            raise RuntimeError("Fluid properties model required")
    
    @Model.build
    def create_matrix(self, mesh):
        """Create momentum matrix system."""
        print(f"[{self.name}] BUILD: Creating momentum matrix...")
        # Use fluid properties for matrix construction
        nu = self.fluid_ref.config.viscosity


# ============================================================================
# Solver
# ============================================================================

class SimpleSolver(BaseModel):
    """Simple incompressible flow solver."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    config: SolverConfig = Field(default_factory=SolverConfig)
    
    # Models
    fluid: FluidPropertiesModel = Field(default_factory=FluidPropertiesModel)
    momentum: MomentumModel = Field(default_factory=MomentumModel)
    
    def get_models(self) -> list:
        """Return all models."""
        return [self.fluid, self.momentum]
    
    @Solver.load
    def load_control_dict(self):
        """Load solver control parameters."""
        print("[solver] LOAD: Loading controlDict...")
        self.config.max_iterations = 50
        self.config.tolerance = 1e-5
    
    @Solver.resolve_dependencies
    def validate_config(self, registry: ModelRegistry):
        """Validate complete configuration."""
        print("[solver] RESOLVE_DEPENDENCIES: Validating configuration...")
        # Check all models are ready
        for model in self.get_models():
            if not hasattr(model, 'fluid_ref') or model.fluid_ref is None:
                if model.name != "fluid":
                    # momentum model should have fluid_ref
                    pass
    
    @Solver.build
    def create_execution_context(self, mesh):
        """Create solver execution context."""
        print("[solver] BUILD: Creating execution context...")


# ============================================================================
# Main Usage Example
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("3-Stage Initialization Example")
    print("="*60)
    print()
    
    # Create solver with models
    solver = SimpleSolver()
    
    print("Stage 1: LOAD")
    print("-" * 60)
    
    # Initialize through 3-stage process
    initializer = SolverInitializer(solver)
    
    # For demonstration, run stages separately
    initializer._run_load()
    print()
    
    print("Stage 2: RESOLVE_DEPENDENCIES")
    print("-" * 60)
    initializer._run_resolve_dependencies()
    print()
    
    print("Stage 3: BUILD")
    print("-" * 60)
    initializer._run_build(mesh=None)
    print()
    
    print("="*60)
    print("Initialization Complete!")
    print("="*60)
    print()
    print(f"Solver configuration:")
    print(f"  - Max iterations: {solver.config.max_iterations}")
    print(f"  - Tolerance: {solver.config.tolerance}")
    print()
    print(f"Fluid properties:")
    print(f"  - Viscosity: {solver.fluid.config.viscosity}")
    print(f"  - Density: {solver.fluid.config.density}")
    print()
    print(f"Models initialized: {[m.name for m in solver.get_models()]}")
    print()
    
    # Or use the convenience method:
    print("Using convenience method initialize():")
    print("-" * 60)
    solver2 = SimpleSolver()
    solver2 = SolverInitializer(solver2).initialize(mesh=None)
    print("Done!")
