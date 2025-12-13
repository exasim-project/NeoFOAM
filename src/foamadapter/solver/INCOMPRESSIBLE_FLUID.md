# IncompressibleFluid Solver

## Overview

The `IncompressibleFluid` solver is a framework-based implementation of the PIMPLE algorithm for incompressible fluid flow simulations. It's a port of the original `pimplefoam.py` solver to the new NeoFOAM framework architecture.

## Architecture

The solver follows the framework's design patterns:

- **Solver Decorator**: Uses `@Solver` to mark the class as a solver
- **Operation Decorator**: Uses `@Solver.operation()` to mark methods as operations
- **Context-based Execution**: All operations work with a shared `Context` object
- **Dependency Management**: Operations specify dependencies using `depends_on` parameter
- **Iterative Operations**: Nested loops (time, PIMPLE, corrector, non-orthogonal) use `IterativeOp`

## Key Components

### Main Class: `IncompressibleFluid`

A Pydantic `BaseModel` that implements the `SolverInterface`:

```python
@Solver
class IncompressibleFluid(BaseModel):
    name: Literal["IncompressibleFluid"] = "IncompressibleFluid"
    argv: list[str] = []
    pRefCell: int | None = None
    pRefValue: float | None = None
    maxDeltaT: float = 1e5
```

### Operations

The solver defines 13 sequential operations:

1. **create_fields**: Reads fields from disk (p, U, phi) and creates turbulence models
2. **setup_models**: Moves PIMPLE control to the models dictionary
3. **print_time**: Outputs current simulation time
4. **momentum_predictor**: Assembles the momentum equation (UEqn)
5. **solve_momentum**: Solves UEqn if momentum predictor is enabled
6. **compute_HbyA**: Computes H/A for the pressure equation
7. **compute_phiHbyA**: Computes flux from H/A
8. **adjust_phi**: Adjusts flux for continuity
9. **solve_pressure**: Solves the pressure Poisson equation
10. **update_flux**: Updates flux after pressure solution
11. **correct_velocity**: Corrects velocity field based on pressure
12. **turbulence_correction**: Updates turbulence model
13. **write_output**: Writes results to disk

### Iterative Conditions

Four condition classes control the nested iteration loops:

- **CFLCondition**: Controls time stepping based on CFL number
- **PimpleLoopCondition**: Controls outer PIMPLE iterations
- **PimpleCorrectorCondition**: Controls pressure corrector iterations
- **NonOrthogonalCondition**: Controls non-orthogonal corrections

## Loop Structure

The solver implements the standard PIMPLE algorithm structure:

```
Time Loop (CFLCondition)
├── Print Time
└── PIMPLE Loop (PimpleLoopCondition)
    ├── Momentum Predictor
    ├── Solve Momentum
    └── Corrector Loop (PimpleCorrectorCondition)
        ├── Compute H/A
        ├── Compute phiHbyA
        ├── Adjust Phi
        └── Non-Orthogonal Loop (NonOrthogonalCondition)
            ├── Solve Pressure
            └── Update Flux
        ├── Correct Velocity
    └── Turbulence Correction
└── Write Output
```

## Usage

### Basic Usage

```python
from foamadapter.solver import IncompressibleFluid

# Create solver with command-line arguments
solver = IncompressibleFluid(argv=["myCase"])

# Run simulation
solver.run()
```

### Advanced: Custom Time Step Control

```python
solver = IncompressibleFluid(
    argv=["myCase"],
    maxDeltaT=0.01  # Maximum time step
)
solver.run()
```

### Integration with Framework

```python
from foamadapter.framework.simulation import Simulation, Domain
from foamadapter.solver import IncompressibleFluid

# Create solver
solver = IncompressibleFluid(argv=["cavity"])

# Create context
ctx = solver.create_context()

# Initialize fields
ops = solver.operations()
ops["create_fields"].run(ctx)
ops["setup_models"].run(ctx)

# Run main loop
solver.main_loop(ctx)
```

## Context Structure

The `Context` object contains:

### Fields Dictionary
- `p`: Pressure field (volScalarField)
- `U`: Velocity field (volVectorField)
- `phi`: Flux field (surfaceScalarField)
- `laminarTransport`: Transport model
- `turbulence`: Turbulence model
- `UEqn`: Momentum equation matrix (during solving)
- `pEqn`: Pressure equation matrix (during solving)
- `rAU`: Inverse of momentum matrix diagonal
- `HbyA`: H/A field for pressure equation
- `phiHbyA`: Flux from H/A

### Models Dictionary
- `pimple`: PIMPLE control object

### Direct Attributes
- `mesh`: Finite volume mesh
- `runTime`: Time control object

## Comparison with Original pimplefoam.py

### Original Structure
```python
class PimpleFoam:
    def momentum_equation(self, ...):
        # All momentum logic in one method
        
    def pressure_correction(self, ...):
        # All pressure logic in one method
        
    def run(self):
        while runTime.loop():
            while pimple.loop():
                UEqn = self.momentum_equation(...)
                while pimple.correct():
                    self.pressure_correction(...)
                if pimple.turbCorr():
                    turbulence.correct()
```

### Framework Structure
```python
@Solver
class IncompressibleFluid:
    @Solver.operation(operation_number=4)
    def momentum_predictor(self, ...):
        # Separate operation
        
    @Solver.operation(operation_number=5)
    def solve_momentum(self, ...):
        # Separate operation
        
    @Solver.operation(operation_number=9)
    def solve_pressure(self, ...):
        # Separate operation
        
    def main_loop(self, ctx):
        # Declarative loop structure using StepBuilder
```

### Benefits of Framework Approach

1. **Modularity**: Each step is a separate, testable operation
2. **Dependency Tracking**: Explicit dependencies between operations
3. **Extensibility**: Easy to add new operations or modify existing ones
4. **Visualization**: Can generate DAG visualizations of the algorithm
5. **Composition**: Operations can be reused in different contexts
6. **Type Safety**: Pydantic validation for configuration
7. **Context Injection**: Automatic injection of required fields into operations

## Testing

A comprehensive test suite is available in `test/framework/test_incompressible_fluid.py`:

- Import tests
- Interface compliance tests
- Operation registration tests
- Dependency verification tests
- Schema validation tests

Run tests with:
```bash
pytest test/framework/test_incompressible_fluid.py -v
```

## Future Extensions

Potential extensions using the framework:

1. **Custom Models**: Add turbulence models as separate Model classes
2. **Multi-Domain**: Extend to handle multiple domains
3. **Custom Operations**: Insert additional operations between existing ones using operation numbers (e.g., 4.1, 4.2)
4. **Coupling**: Interface with other solvers through shared context
5. **Monitoring**: Add operations for runtime monitoring and adaptation

## References

- Original implementation: `src/foamadapter/solver/pimplefoam.py`
- Framework documentation: `src/foamadapter/framework/`
- Test examples: `test/framework/integration/`
