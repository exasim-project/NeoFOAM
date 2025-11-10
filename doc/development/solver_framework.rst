================
Solver Framework
================

Introduction
============

General Idea
------------

CFD solvers are essentially a **list of steps that update fields** (velocity, pressure, temperature, etc.). The Solver Framework provides a declarative way to define these steps and their dependencies.

**Core Concept:**

* CFD solver = sequence of computational steps
* Each step updates physical fields
* Steps have dependencies (e.g., "solve pressure" depends on "solve momentum")
* The framework handles the execution order automatically

**Key Components:**

1. **Solver**: Main computational component containing steps
2. **Model**: Reusable computational module (can be nested in solvers)
3. **Operations**: Executable steps that update fields
4. **Context**: Container holding fields and models
5. **DAG**: Automatically orders operations based on dependencies

The framework converts your step declarations into operations, builds a dependency graph (DAG), and executes them in the correct order.


Why Use This Framework?
------------------------

**Traditional Approach:**

.. code-block:: python

    def solve():
        initialize()
        update_boundary_conditions()
        solve_momentum()
        solve_pressure()
        correct_velocity()
        update_turbulence()
        # Order is hard-coded and fragile

**Framework Approach:**

.. code-block:: python

    @Solver
    class MySolver:
        @Solver.step(step_number=1)
        def initialize(self, U, p): ...
        
        @Solver.step(step_number=3, depends_on=["solve_momentum"])
        def solve_pressure(self, U, p): ...
        
        # Framework handles the ordering automatically!

**Benefits:**

* Declarative: Just declare steps and dependencies
* Automatic ordering: DAG handles execution order
* Reusable: Models can be shared across solvers
* Testable: Each step can be tested independently
* Visualizable: See your solver structure as a graph


Framework Architecture
======================

How It Works
------------

.. code-block:: text

    ┌─────────────┐
    │   Solver    │  Define steps with @Solver.step
    │  or Model   │  Specify dependencies
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Operations  │  Steps converted to Operations
    │ Collection  │  Stored in OperationCollection
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │     DAG     │  Build dependency graph
    │  (Graph)    │  Topological sort
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Execute    │  Run operations in order
    │  with       │  Pass Context to each step
    │  Context    │  Update fields
    └─────────────┘

**Flow:**

1. Define solver/model with decorated steps
2. Call ``operations()`` to collect all steps as Operation objects
3. Build DAG from operation dependencies
4. Execute operations in topological order
5. Each operation receives Context and updates fields


Core Components
---------------

Solver
^^^^^^

A **Solver** is the main computational component that defines how to solve your problem.

.. code-block:: python

    from foamadapter.framework.solver import Solver
    from foamadapter.framework.context import Context, FieldUpdates
    
    @Solver
    class MyCustomSolver(BaseModel):
        name: str = "MyCustomSolver"
        
        def create_context(self) -> Context:
            """Initialize the context with fields and models."""
            return Context(
                fields={"U": ..., "p": ...},
                models={}
            )
        
        @Solver.step(step_number=1)
        def initialize(self, U, p) -> FieldUpdates:
            # Update fields
            return FieldUpdates({"U": U, "p": p})
        
        def operations(self, domain_name: str | None = None):
            """Collect all decorated steps as operations."""
            funcs = decorated_member_functions(self)
            ops = OperationCollection()
            for func in funcs:
                ops.add(Operation.create_SeqOp(func))
            return ops
        
        def main_loop(self, ctx: Context):
            """Execute all operations."""
            ops = Operations(self.operations())
            ops.run(ctx)

**Required Methods:**

* ``operations()``: Returns OperationCollection of all steps
* ``main_loop()``: Defines how to execute the operations


Model
^^^^^

A **Model** is a reusable computational module that can be embedded in solvers.

.. code-block:: python

    from foamadapter.framework.model import Model
    
    @Model
    class PressureCorrectionModel(BaseModel):
        max_iterations: int = 50
        
        @Model.step(step_number=1)
        def compute_pressure_correction(self, p, U) -> FieldUpdates:
            # Compute correction
            return FieldUpdates({"p": p_corrected})
        
        def operations(self) -> OperationCollection:
            """Same as Solver."""
            funcs = decorated_member_functions(self)
            ops = OperationCollection()
            for func in funcs:
                ops.add(Operation.create_SeqOp(func))
            return ops
        
        def run(self, ctx: Context):
            """Execute model operations."""
            ops = Operations(self.operations())
            ops.run(ctx)

**Difference from Solver:**

* Models are reusable components
* Models are typically called from within solver steps
* Models don't have ``create_context()`` or ``main_loop()``
* Models have ``run()`` method


Operations
^^^^^^^^^^

**Operations** are the executable units created from your decorated steps.

.. code-block:: python

    # Your decorated method
    @Solver.step(step_number=2, depends_on=["initialize"])
    def solve_momentum(self, U, p) -> FieldUpdates:
        # ... computation
        return FieldUpdates({"U": U_new})
    
    # Framework converts this to an Operation:
    operation = Operation(
        func=SequentialOp(solve_momentum),
        step_name="solve_momentum",
        step_number=2,
        depends_on=["initialize"]
    )

**Operation Types:**

* ``SequentialOp``: Regular step (most common)
* ``ConditionalOp``: Step with condition (for if-statements)
* ``IterativeOp``: Step with loop (for while-loops)


Context
^^^^^^^

**Context** holds all data passed between operations.

.. code-block:: python

    from foamadapter.framework.context import Context, FieldUpdates
    
    # Create context
    ctx = Context(
        fields={
            "U": velocity_field,
            "p": pressure_field,
            "T": temperature_field
        },
        models={
            "turbulence": turbulence_model,
            "transport": transport_properties
        }
    )
    
    # Steps automatically get parameters from context by name
    @Solver.step(step_number=1)
    def my_step(self, U, p):  # U and p injected from ctx.fields
        # Modify fields
        U_new = U * 2
        p_new = p + 1
        
        # Return updates
        return FieldUpdates({"U": U_new, "p": p_new})

**Context Attributes:**

* ``fields``: Physical quantities (U, p, T, etc.)
* ``models``: Computational models
* ``mesh``: Mesh object (optional)
* ``runTime``: Time management (optional)


Creating Your First Solver
===========================

Step-by-Step Guide
------------------

**Step 1: Import Required Components**

.. code-block:: python

    from typing import Literal
    from pydantic import BaseModel
    from foamadapter.framework.solver import Solver
    from foamadapter.framework.context import Context, FieldUpdates
    from foamadapter.framework.decorator import decorated_member_functions
    from foamadapter.framework.operations import Operation, OperationCollection, Operations

**Step 2: Define Your Solver Class**

.. code-block:: python

    @Solver
    class SimpleSolver(BaseModel):
        name: Literal["SimpleSolver"] = "SimpleSolver"
        tolerance: float = 1e-6
        max_iterations: int = 100

**Step 3: Define Computational Steps**

.. code-block:: python

    @Solver
    class SimpleSolver(BaseModel):
        name: Literal["SimpleSolver"] = "SimpleSolver"
        
        @Solver.step(step_number=1)
        def initialize_fields(self, U, p) -> FieldUpdates:
            """Initialize velocity and pressure."""
            U = U * 0  # Zero velocity
            p = p * 0  # Zero pressure
            return FieldUpdates({"U": U, "p": p})
        
        @Solver.step(step_number=2, depends_on=["initialize_fields"])
        def solve_momentum(self, U, p) -> FieldUpdates:
            """Solve momentum equation."""
            # Your momentum equation solver
            U_new = self._solve_momentum_equation(U, p)
            return FieldUpdates({"U": U_new})
        
        @Solver.step(step_number=3, depends_on=["solve_momentum"])
        def solve_pressure(self, U, p) -> FieldUpdates:
            """Solve pressure equation."""
            # Your pressure equation solver
            p_new = self._solve_pressure_equation(U, p)
            return FieldUpdates({"p": p_new})
        
        @Solver.step(step_number=4, depends_on=["solve_pressure"])
        def correct_velocity(self, U, p) -> FieldUpdates:
            """Correct velocity with pressure gradient."""
            U_corrected = U - self._compute_pressure_gradient(p)
            return FieldUpdates({"U": U_corrected})

**Step 4: Implement Required Methods**

.. code-block:: python

    @Solver
    class SimpleSolver(BaseModel):
        # ... steps defined above ...
        
        def create_context(self) -> Context:
            """Initialize context with fields."""
            return Context(
                fields={
                    "U": self._initialize_velocity(),
                    "p": self._initialize_pressure()
                },
                models={}
            )
        
        def operations(self, domain_name: str | None = None) -> OperationCollection:
            """Collect all decorated steps."""
            funcs = decorated_member_functions(self)
            ops = OperationCollection()
            for func in funcs:
                op = Operation.create_SeqOp(func, domain_name=domain_name)
                ops.add(op)
            return ops
        
        def main_loop(self, ctx: Context):
            """Execute solver steps."""
            ops = Operations(self.operations())
            ops.run(ctx)

**Step 5: Use Your Solver**

.. code-block:: python

    # Create solver instance
    solver = SimpleSolver(tolerance=1e-6, max_iterations=100)
    
    # Create context
    ctx = solver.create_context()
    
    # Run solver
    solver.main_loop(ctx)
    
    # Access results
    final_velocity = ctx.fields["U"]
    final_pressure = ctx.fields["p"]


Understanding the @step Decorator
----------------------------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

    @Solver.step(step_number=1)
    def my_step(self, U, p) -> FieldUpdates:
        # Computation
        return FieldUpdates({"U": U_new, "p": p_new})

**Components:**

* ``step_number``: Order hint (not strict ordering, just priority)
* Function parameters (``U``, ``p``): Automatically injected from ``context.fields``
* Return ``FieldUpdates``: Dictionary of fields to update in context


Step Numbering
^^^^^^^^^^^^^^

Step numbers provide a **hint** for ordering, but the DAG uses dependencies for final order.

.. code-block:: python

    @Solver.step(step_number=1)
    def first(self, U): ...
    
    @Solver.step(step_number=2)
    def second(self, U): ...
    
    # Executed in order: first → second

**Hierarchical Numbering:**

.. code-block:: python

    @Solver.step(step_number="1")
    def main_step(self, U): ...
    
    @Solver.step(step_number="1.1", depends_on=["main_step"])
    def sub_step_1(self, U): ...
    
    @Solver.step(step_number="1.2", depends_on=["sub_step_1"])
    def sub_step_2(self, U): ...
    
    # Executed: main_step → sub_step_1 → sub_step_2


Dependencies
^^^^^^^^^^^^

Explicit dependencies ensure correct execution order:

.. code-block:: python

    @Solver.step(step_number=1)
    def solve_momentum(self, U): ...
    
    @Solver.step(step_number=2, depends_on=["solve_momentum"])
    def solve_pressure(self, U, p): ...
    
    @Solver.step(step_number=3, depends_on=["solve_pressure"])
    def correct_velocity(self, U, p): ...
    
    # DAG ensures: solve_momentum → solve_pressure → correct_velocity

**Multiple Dependencies:**

.. code-block:: python

    @Solver.step(step_number=3, depends_on=["solve_momentum", "solve_energy"])
    def couple_fields(self, U, T): ...


Parameter Injection
^^^^^^^^^^^^^^^^^^^

The framework automatically injects parameters from context:

.. code-block:: python

    # Context has:
    ctx = Context(
        fields={"U": velocity, "p": pressure, "T": temperature},
        models={"turbulence": turb_model}
    )
    
    # Simple injection
    @Solver.step(step_number=1)
    def step1(self, U, p):  # Gets ctx.fields["U"], ctx.fields["p"]
        ...
    
    # Inject entire context
    @Solver.step(step_number=2)
    def step2(self, ctx: Context):  # Gets the context itself
        U = ctx.fields["U"]
        ...
    
    # Annotated types for clarity
    from foamadapter.framework.context import Field, Model
    from typing import Annotated
    
    @Solver.step(step_number=3)
    def step3(
        self,
        U: Field[object],  # From ctx.fields
        turbulence: Model[object]  # From ctx.models
    ):
        ...


Complete Example: SIMPLE Solver
--------------------------------

Here's a complete example of a SIMPLE-like solver:

.. code-block:: python

    from typing import Literal
    from pydantic import BaseModel
    from foamadapter.framework.solver import Solver
    from foamadapter.framework.context import Context, FieldUpdates
    from foamadapter.framework.decorator import decorated_member_functions
    from foamadapter.framework.operations import Operation, OperationCollection, Operations
    
    @Solver
    class SIMPLESolver(BaseModel):
        """SIMPLE algorithm for pressure-velocity coupling."""
        
        name: Literal["SIMPLESolver"] = "SIMPLESolver"
        n_correctors: int = 3
        tolerance: float = 1e-6
        
        def create_context(self) -> Context:
            """Initialize fields."""
            return Context(
                fields={
                    "U": self._create_velocity_field(),
                    "p": self._create_pressure_field(),
                    "phi": self._create_flux_field()
                },
                models={}
            )
        
        @Solver.step(step_number=1)
        def momentum_predictor(self, U, p, phi) -> FieldUpdates:
            """Solve momentum equation with old pressure."""
            # Solve: ∂U/∂t + ∇·(UU) = -∇p + ∇·(ν∇U)
            U_star = self._solve_momentum(U, p, phi)
            return FieldUpdates({"U": U_star})
        
        @Solver.step(step_number=2, depends_on=["momentum_predictor"])
        def pressure_equation(self, U, p, phi) -> FieldUpdates:
            """Solve pressure equation for correction."""
            # Solve: ∇²p = ∇·U*
            p_new = self._solve_pressure_poisson(U, p)
            return FieldUpdates({"p": p_new})
        
        @Solver.step(step_number=3, depends_on=["pressure_equation"])
        def velocity_corrector(self, U, p) -> FieldUpdates:
            """Correct velocity with new pressure gradient."""
            # U = U* - ∇p
            U_corrected = U - self._compute_grad_p(p)
            return FieldUpdates({"U": U_corrected})
        
        @Solver.step(step_number=4, depends_on=["velocity_corrector"])
        def flux_corrector(self, U, phi) -> FieldUpdates:
            """Update face fluxes."""
            phi_new = self._interpolate_flux(U)
            return FieldUpdates({"phi": phi_new})
        
        def operations(self, domain_name: str | None = None) -> OperationCollection:
            """Collect all steps as operations."""
            funcs = decorated_member_functions(self)
            ops = OperationCollection()
            for func in funcs:
                op = Operation.create_SeqOp(func, domain_name=domain_name)
                ops.add(op)
            return ops
        
        def main_loop(self, ctx: Context):
            """Run SIMPLE iterations."""
            ops = Operations(self.operations())
            
            for iteration in range(self.n_correctors):
                ops.run(ctx)
                
                if self._check_convergence(ctx):
                    break
        
        def _solve_momentum(self, U, p, phi):
            # Implementation
            pass
        
        def _solve_pressure_poisson(self, U, p):
            # Implementation
            pass
        
        def _compute_grad_p(self, p):
            # Implementation
            pass
    
    # Usage
    solver = SIMPLESolver(n_correctors=50, tolerance=1e-6)
    ctx = solver.create_context()
    solver.main_loop(ctx)


Creating Reusable Models
=========================

What is a Model?
----------------

A **Model** is a reusable computational component that can be embedded in solvers or other models.

**When to Use Models:**

* Reusable algorithms (turbulence models, transport properties)
* Sub-iterations (PISO corrector loops, iterative solvers)
* Modular components (equation systems, correction steps)

**Solver vs Model:**

+------------------------+----------------------------------+
| Solver                 | Model                            |
+========================+==================================+
| Top-level component    | Nested component                 |
+------------------------+----------------------------------+
| Has ``main_loop()``    | Has ``run()``                    |
+------------------------+----------------------------------+
| Creates context        | Uses existing context            |
+------------------------+----------------------------------+
| Entry point            | Called from solver/model steps   |
+------------------------+----------------------------------+


Building a Model
----------------

**Step 1: Define Model Class**

.. code-block:: python

    from foamadapter.framework.model import Model
    
    @Model
    class PISOCorrectorModel(BaseModel):
        """PISO corrector loop for pressure-velocity coupling."""
        
        name: Literal["PISOCorrector"] = "PISOCorrector"
        n_correctors: int = 2
        current_iteration: int = 0

**Step 2: Add Model Steps**

.. code-block:: python

    @Model
    class PISOCorrectorModel(BaseModel):
        n_correctors: int = 2
        current_iteration: int = 0
        
        def loop_condition(self) -> bool:
            """Check if more corrector iterations needed."""
            self.current_iteration += 1
            return self.current_iteration <= self.n_correctors
        
        @Model.step(step_number=1)
        def solve_pressure(self, U, p) -> FieldUpdates:
            """Pressure correction step."""
            p_corrected = self._pressure_equation(U, p)
            return FieldUpdates({"p": p_corrected})
        
        @Model.step(step_number=2, depends_on=["solve_pressure"])
        def correct_velocity(self, U, p) -> FieldUpdates:
            """Velocity correction step."""
            U_corrected = U - self._grad_p(p)
            return FieldUpdates({"U": U_corrected})

**Step 3: Implement Required Methods**

.. code-block:: python

    @Model
    class PISOCorrectorModel(BaseModel):
        # ... steps above ...
        
        def operations(self) -> OperationCollection:
            """Collect model operations."""
            funcs = decorated_member_functions(self)
            ops = OperationCollection()
            for func in funcs:
                op = Operation.create_SeqOp(func)
                ops.add(op)
            return ops
        
        def run(self, ctx: Context):
            """Execute corrector loop."""
            ops = Operations(self.operations())
            
            # Reset counter
            self.current_iteration = 0
            
            # Run corrector loop
            while self.loop_condition():
                ops.run(ctx)


Integrating Models into Solvers
--------------------------------

Models can be called from solver steps:

.. code-block:: python

    @Solver
    class PISOSolver(BaseModel):
        """PISO solver with corrector model."""
        
        name: str = "PISOSolver"
        corrector: PISOCorrectorModel
        
        @Solver.step(step_number=1)
        def momentum_predictor(self, U, p) -> FieldUpdates:
            """Solve momentum equation."""
            U_star = self._solve_momentum(U, p)
            return FieldUpdates({"U": U_star})
        
        @Solver.step(step_number=2, depends_on=["momentum_predictor"])
        def piso_correctors(self, ctx: Context):
            """Run PISO corrector loop."""
            # Call model with context
            self.corrector.run(ctx)
        
        @Solver.step(step_number=3, depends_on=["piso_correctors"])
        def update_properties(self, U, p) -> FieldUpdates:
            """Update derived quantities."""
            phi = self._compute_flux(U)
            return FieldUpdates({"phi": phi})
        
        def operations(self, domain_name: str | None = None) -> OperationCollection:
            funcs = decorated_member_functions(self)
            ops = OperationCollection()
            for func in funcs:
                op = Operation.create_SeqOp(func, domain_name=domain_name)
                ops.add(op)
            return ops
        
        def main_loop(self, ctx: Context):
            ops = Operations(self.operations())
            ops.run(ctx)
    
    # Usage
    corrector = PISOCorrectorModel(n_correctors=2)
    solver = PISOSolver(corrector=corrector)
    ctx = solver.create_context()
    solver.main_loop(ctx)


Complete Model Example: Iterative Sub-Solver
---------------------------------------------

.. code-block:: python

    @Model
    class IterativeLinearSolver(BaseModel):
        """Iterative solver model (e.g., for linear systems)."""
        
        name: str = "IterativeLinearSolver"
        max_iterations: int = 100
        tolerance: float = 1e-6
        current_iteration: int = 0
        residual: float = 1.0
        
        def converged(self) -> bool:
            """Check convergence."""
            return self.residual < self.tolerance
        
        def max_iterations_reached(self) -> bool:
            """Check iteration limit."""
            return self.current_iteration >= self.max_iterations
        
        def should_continue(self) -> bool:
            """Loop condition."""
            return not self.converged() and not self.max_iterations_reached()
        
        @Model.step(step_number=1)
        def compute_residual(self, A, x, b) -> FieldUpdates:
            """Compute residual: r = b - Ax."""
            r = b - A @ x
            self.residual = self._norm(r)
            return FieldUpdates({"r": r})
        
        @Model.step(step_number=2, depends_on=["compute_residual"])
        def update_solution(self, x, r) -> FieldUpdates:
            """Update solution."""
            x_new = x + self._compute_correction(r)
            self.current_iteration += 1
            return FieldUpdates({"x": x_new})
        
        def operations(self) -> OperationCollection:
            funcs = decorated_member_functions(self)
            ops = OperationCollection()
            for func in funcs:
                op = Operation.create_SeqOp(func)
                ops.add(op)
            return ops
        
        def run(self, ctx: Context):
            """Run iterative solver."""
            ops = Operations(self.operations())
            
            # Reset state
            self.current_iteration = 0
            self.residual = 1.0
            
            # Iterative loop
            while self.should_continue():
                ops.run(ctx)


Dependency Management and DAG
==============================

How the DAG Works
-----------------

The framework builds a **Directed Acyclic Graph (DAG)** from your operations:

1. Each step becomes a **node** in the graph
2. Each dependency becomes an **edge** in the graph
3. The DAG is **topologically sorted** to find execution order
4. Step numbers provide **priority hints** for sorting

.. code-block:: python

    # Your steps:
    @Solver.step(step_number=1)
    def step_A(self): ...
    
    @Solver.step(step_number=2, depends_on=["step_A"])
    def step_B(self): ...
    
    @Solver.step(step_number=3, depends_on=["step_B"])
    def step_C(self): ...
    
    # Results in DAG:
    # step_A → step_B → step_C
    
    # Execution order: [step_A, step_B, step_C]


Declaring Dependencies
----------------------

**Explicit Dependencies:**

.. code-block:: python

    @Solver.step(step_number=2, depends_on=["initialize"])
    def solve_momentum(self, U): ...
    
    @Solver.step(step_number=3, depends_on=["solve_momentum", "update_bc"])
    def solve_pressure(self, U, p): ...

**Multiple Independent Steps:**

.. code-block:: python

    @Solver.step(step_number=1)
    def initialize(self): ...
    
    # These can run in any order after initialize
    @Solver.step(step_number=2, depends_on=["initialize"])
    def setup_momentum(self): ...
    
    @Solver.step(step_number=2, depends_on=["initialize"])
    def setup_energy(self): ...
    
    # This needs both
    @Solver.step(step_number=3, depends_on=["setup_momentum", "setup_energy"])
    def couple_equations(self): ...


Visualizing Your Solver
------------------------

Generate an interactive HTML visualization of your solver's DAG:

.. code-block:: python

    from foamadapter.framework.pyvis_utils import digraph_to_pyvis_html
    from foamadapter.framework.dag import build_dag
    
    # Create solver
    solver = MySolver()
    
    # Get operations
    ops = solver.operations()
    
    # Build DAG
    metadata = [op.operation_metadata() for op in ops.ops]
    dag = build_dag(metadata)
    
    # Visualize
    digraph_to_pyvis_html(dag, "solver_dag.html")
    
    # Open solver_dag.html in browser to see interactive graph

**DAG Visualization Features:**

* Nodes represent operations
* Edges show dependencies
* Colors indicate operation types
* Hover for details
* Interactive layout


Running Simulations
===================

Complete Example
----------------

Here's how to set up and run a complete simulation:

.. code-block:: python

    from foamadapter.framework.simulation import Simulation, Domain
    from foamadapter.framework.solver import Solver
    from foamadapter.framework.context import Context, FieldUpdates
    
    # Define your solver (as shown above)
    @Solver
    class MySolver(BaseModel):
        # ... solver implementation ...
        pass
    
    # Create simulation with domain
    simulation = Simulation(
        domains=[
            Domain(
                name="region1",
                solver=MySolver()
            )
        ],
        coupling_interface=[]  # For future multi-domain coupling
    )
    
    # Initialize simulation context
    sim_ctx = simulation.init_simulation_context()
    
    # Run simulation
    simulation.main_loop(sim_ctx)
    
    # Access results
    region1_ctx = sim_ctx.domain_context["region1"]
    final_velocity = region1_ctx.fields["U"]
    final_pressure = region1_ctx.fields["p"]


Multi-Domain Simulations (Future)
----------------------------------

The framework supports multiple domains with different solvers:

.. code-block:: python

    simulation = Simulation(
        domains=[
            Domain(name="fluid", solver=FluidSolver()),
            Domain(name="solid", solver=SolidSolver())
        ],
        coupling_interface=[
            # Define coupling between domains
        ]
    )
    
    # Framework will handle inter-domain dependencies
    dag = simulation.dependency_graph()


Best Practices
==============

Solver Design
-------------

**1. Single Responsibility**

Each step should do one thing:

.. code-block:: python

    # Good
    @Solver.step(step_number=1)
    def solve_momentum(self, U, p): ...
    
    @Solver.step(step_number=2)
    def solve_pressure(self, U, p): ...
    
    # Avoid
    @Solver.step(step_number=1)
    def solve_everything(self, U, p, T, k, epsilon): ...

**2. Clear Dependencies**

Always specify dependencies explicitly:

.. code-block:: python

    # Good
    @Solver.step(step_number=2, depends_on=["solve_momentum"])
    def solve_pressure(self, U, p): ...
    
    # Avoid relying only on step numbers
    @Solver.step(step_number=2)
    def solve_pressure(self, U, p): ...  # Unclear dependency

**3. Meaningful Step Numbers**

Use hierarchical numbering for related steps:

.. code-block:: python

    @Solver.step(step_number="1")
    def momentum_predictor(self): ...
    
    @Solver.step(step_number="1.1")
    def momentum_sub_step_1(self): ...
    
    @Solver.step(step_number="1.2")
    def momentum_sub_step_2(self): ...
    
    @Solver.step(step_number="2")
    def pressure_corrector(self): ...


Naming Conventions
------------------

**Step Names:**

* Use descriptive verb phrases: ``solve_momentum``, ``update_boundary_conditions``
* Avoid abbreviations: ``initialize`` not ``init``
* Use snake_case: ``solve_pressure`` not ``solvePressure``

**Field Names:**

* Use standard CFD symbols: ``U`` (velocity), ``p`` (pressure), ``T`` (temperature)
* Use descriptive names: ``turbulent_viscosity`` not ``nut``
* Be consistent across solvers

**Model Names:**

* Use descriptive names: ``PISOCorrectorModel``, ``TurbulenceModel``
* Suffix with ``Model``: ``PressureCorrectionModel``


Common Pitfalls
---------------

**1. Circular Dependencies**

.. code-block:: python

    # ERROR: Circular dependency
    @Solver.step(step_number=1, depends_on=["step_B"])
    def step_A(self): ...
    
    @Solver.step(step_number=2, depends_on=["step_A"])
    def step_B(self): ...
    
    # Solution: Remove circular dependency
    @Solver.step(step_number=1)
    def step_A(self): ...
    
    @Solver.step(step_number=2, depends_on=["step_A"])
    def step_B(self): ...

**2. Missing Context Fields**

.. code-block:: python

    # ERROR: Field not in context
    @Solver.step(step_number=1)
    def my_step(self, U, nonexistent_field):  # KeyError!
        ...
    
    # Solution: Ensure field exists in context
    def create_context(self) -> Context:
        return Context(
            fields={
                "U": ...,
                "nonexistent_field": ...  # Add missing field
            },
            models={}
        )

**3. Forgetting to Return FieldUpdates**

.. code-block:: python

    # ERROR: No return value
    @Solver.step(step_number=1)
    def my_step(self, U):
        U = U * 2  # Context not updated!
    
    # Solution: Return FieldUpdates
    @Solver.step(step_number=1)
    def my_step(self, U):
        U = U * 2
        return FieldUpdates({"U": U})


API Quick Reference
===================

Decorators
----------

.. code-block:: python

    # Solver decorator
    @Solver
    class MySolver(BaseModel): ...
    
    # Model decorator
    @Model
    class MyModel(BaseModel): ...
    
    # Step decorator
    @Solver.step(step_number=1, depends_on=["other_step"])
    def my_step(self, U, p) -> FieldUpdates: ...
    
    # Condition decorator (for future use)
    @Solver.condition(step_number=1)
    def check_convergence(self, residual) -> bool: ...


Core Classes
------------

.. code-block:: python

    # Context
    from foamadapter.framework.context import Context, FieldUpdates
    
    ctx = Context(
        fields={"U": ..., "p": ...},
        models={"turbulence": ...}
    )
    
    # Update fields
    return FieldUpdates({"U": U_new, "p": p_new})
    
    # Operations
    from foamadapter.framework.operations import (
        Operation,
        OperationCollection,
        Operations
    )
    
    ops = OperationCollection()
    ops.add(operation)
    
    # Domain and Simulation
    from foamadapter.framework.simulation import Domain, Simulation
    
    domain = Domain(name="region1", solver=MySolver())
    simulation = Simulation(domains=[domain], coupling_interface=[])


Required Methods
----------------

**Solver Interface:**

.. code-block:: python

    class MySolver:
        def create_context(self) -> Context:
            """Initialize context with fields and models."""
            ...
        
        def operations(self, domain_name: str | None = None) -> OperationCollection:
            """Collect all decorated steps as operations."""
            funcs = decorated_member_functions(self)
            ops = OperationCollection()
            for func in funcs:
                ops.add(Operation.create_SeqOp(func, domain_name=domain_name))
            return ops
        
        def main_loop(self, ctx: Context):
            """Execute solver."""
            ops = Operations(self.operations())
            ops.run(ctx)

**Model Interface:**

.. code-block:: python

    class MyModel:
        def operations(self) -> OperationCollection:
            """Collect model operations."""
            funcs = decorated_member_functions(self)
            ops = OperationCollection()
            for func in funcs:
                ops.add(Operation.create_SeqOp(func))
            return ops
        
        def run(self, ctx: Context):
            """Execute model."""
            ops = Operations(self.operations())
            ops.run(ctx)


Utility Functions
-----------------

.. code-block:: python

    # Collect decorated methods
    from foamadapter.framework.decorator import decorated_member_functions
    
    funcs = decorated_member_functions(solver_instance)
    
    # Build DAG
    from foamadapter.framework.dag import build_dag, compute_steps_order
    
    dag = build_dag(operation_metadata_list)
    ordered_ops = compute_steps_order(operation_collection)
    
    # Visualize
    from foamadapter.framework.pyvis_utils import digraph_to_pyvis_html
    
    digraph_to_pyvis_html(dag, "output.html")


Summary
=======

The Solver Framework provides a declarative way to define CFD solvers:

1. **Define steps** with ``@Solver.step`` or ``@Model.step``
2. **Specify dependencies** with ``depends_on`` parameter
3. **Return updates** with ``FieldUpdates``
4. **Framework handles** operation ordering via DAG
5. **Reuse models** across different solvers

**Key Benefits:**

* Automatic dependency resolution
* Clear, testable code structure
* Reusable models and components
* Visual representation of solver structure
* Type-safe parameter injection

Start by creating a simple solver, then gradually add complexity as needed. The framework grows with your needs!
