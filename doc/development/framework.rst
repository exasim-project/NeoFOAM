FoamAdapter Framework
=====================

Overview
--------

FoamAdapter makes it easy to build and extend scientific solvers in Python.

FoamAdapter Framework
=====================

Overview
--------

FoamAdapter makes it easy to build and extend scientific solvers in Python.
The core idea is simple:
A solver is just a list of steps that update fields (variables) in your simulation.
Each solver is specific to a domain (like a region or a physical model), and you can extend it with models that add or change steps at runtime.

**How to read the diagram below:**
- The **Main Solver Loop** runs repeatedly, controlled by a main loop condition (e.g., convergence or max iterations).
- Each **Step** (e.g., Momentum, Energy) updates the simulation fields.
- Some steps, like **Pressure Iteration**, contain their own inner loop (e.g., the PISO loop), with their own condition.
- **Models** can inject new steps or conditions into the solver at runtime, making the solver flexible and extensible.
- **Fields** are updated by the solver steps and influence the next iteration.

**Key Concepts:**
- **Steps** are modular actions that update fields.
- **Iterations** are controlled by conditions, which can be injected or modified by models.
- **Models** extend solvers by adding steps or conditions, without changing the core solver code.
- **Nested loops** (like the PISO loop) are supported by defining steps that themselves contain iteration logic.

Visual Overview
---------------

The following diagram shows how a typical solver is structured in FoamAdapter, including steps, iterations, conditions, and model extensions:

.. mermaid::

	flowchart TD
		subgraph Solver["Main Solver Loop"]
			direction TB
			Loop{"Main Loop Condition"}
			Step1["Step 1: Momentum"]
			Step2["Step 2: Energy"]
			subgraph Step3["Step 3: Pressure Iteration"]
				direction TB
				Condition{"PISO LOOP"}
				Step3a["Step 3.1: Assumble System"]
				Step3b["Step 3.2: Update Pressure"]
				Condition --> Step3a
				Step3a --> Step3b
				Step3b -- "check condition" --> Condition
			end
			Step4["Step 4: Turbulence Update"]
		end
		subgraph Model["Model"]
			direction TB
			ModelS1["Residual Calculation"]
			ModelS2["Energy"]

		end
		Fields["Simulation Fields"]
		Fields -- "modified by" --> Loop
		Loop --> Step1
		Step1 --> Step2
		Step2 --> Step3
		Step3 --> Step4

		ModelS1 -- "injects condition" --> Loop
		ModelS2 -- "injects step" --> Step2

		style Model fill:#E3F2FD

**How to read the diagram above:**
- The **Main Solver Loop** runs repeatedly, controlled by a main loop condition (e.g., convergence or max iterations).
- Each **Step** (e.g., Momentum, Energy) updates the simulation fields.
- Some steps, like **Pressure Iteration**, contain their own inner loop (e.g., the PISO loop), with their own condition.
- **Models** can inject new steps or conditions into the solver at runtime, making the solver flexible and extensible.
- **Fields** are updated by the solver steps and influence the next iteration.

**Key Concepts:**
- **Steps** are modular actions that update fields.
- **Iterations or nested loops** are controlled by conditions, which can be injected or modified by models (like the PISO loop)
- **Models** extend solvers by adding steps or conditions, without changing the core solver code.

In practice, you build a solver by listing the steps you want to run, and you can add more steps or conditions by attaching models.
Each step updates the simulation fields, and the solver runs in a loop until a stopping condition is met.
Some steps can themselves contain inner loops (like the PISO pressure iteration), and models can inject new steps or conditions anywhere in the process.

.. code-block:: python

	class MySolver(BaseModel):

		@Solver.step(step_number=1)
		def step_1(self, ctx):
			# first step
			...
		@Solver.step(step_number=2)
		def step_2(self, ctx):
			# second step
			...

        def steps(self):
            ...

        def main_loop(self, ctx):
                for step in self.steps():
                    step.run(ctx)

Steps
-----

A step is a single action in a solver, model or iteration.
Steps are the building blocks for a component and are used as decorators: ``@[Component].step``:


.. code-block:: python

    @Solver
    class MySolver(BaseModel):

        @Solver.step(step_number=1)
        def update_temperature(self, temperature: float):
            # Use the temperature field and context
            ...

    @Model
    class MyModel(BaseModel):

        @Model.step(step_number=2)
        def compute_residual(self, pressure: float):
            # Use the pressure field and context
            ...

You can control the order of steps and specify dependencies between them.
When a step runs, FoamAdapter automatically gives it the fields it needs from the simulation context.


Features Implementations
~~~~~~~~~~~~~~~~~~~~~~~~


The core of the FoamAdapter framework's modularity is the `step` decorator.

It allows to define steps in Solvers, Models, and Iterations in a consistent way and 

* registers them with metadata (like order and dependencies).
* automatically injects required fields from the simulation context when the step is run.
* enables to specify dependencies between steps, allowing for complex execution order.

.. admonition:: Framework NBI Dataclasses Test (Click to expand)
   :class: toggle

    .. literalinclude:: ../../src/foamadapter/framework/step.py
        :language: python
        :pyobject: _step
        :caption: step decorator implementation

How the step decorator works:
""""""""""""""""""""""""""""""

The `step` decorator is defined in `foamadapter.framework.step`.
When applied to a method, it wraps the method to add metadata and handle field injection.
This is achieved by attaching different attributes to the method that are specified on the function parameter annotations and the return type annotation.

Additionally, the decorator attaches a `run` method to the decorated function.
This `run` method is responsible for executing the step with automatic field injection from the provided simulation context.

.. code-block:: python

    @Solver
    class MySolver(BaseModel):

        @Solver.step(step_number=1, depends_on=["previous_step"])
        def my_step(field_a: float, field_b: int) -> FieldUpdates:
            # Step implementation
            ...

    MySolver.my_step.run(ctx) # <-- This will inject field_a and field_b from ctx.fields

Iterations
----------

Most solvers need to repeat their steps until some condition is met (like convergence or a max number of steps).
FoamAdapter lets you control this with a condition object.
You can use simple conditions (like a maximum number of iterations) or combine them for more complex logic.

Example Iteration Loop
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

	while not condition():
		for step in solver.steps:
			step.run(ctx)

What is a Condition?
--------------------

A condition tells the solver when to stop iterating.
You can use built-in conditions (like max iterations), or combine them with ``&`` (and), ``|`` (or), and ``~`` (not) to make your own.

What is a Model?
----------------

Models are plug-ins that can add new steps or change how a solver works.
They are especially useful for adding new physics or features without rewriting your solver.
When you attach a model to a solver, FoamAdapter figures out where to insert the model’s steps by solving a dependency graph (DAG).
Models can also add new conditions or change the iteration logic.

Example: Model Injecting a Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

	class MyModel(BaseModel):
		@Solver.step(step_number=3, depends_on=["solve"])
		def postprocess(self, ctx):
			# Do something after solve
			...

	# At runtime, MyModel.postprocess is added to the solver’s steps automatically.

Summary
-------

FoamAdapter helps you build flexible, modular solvers by breaking them into steps and letting you extend them with models and conditions.
You focus on the science; FoamAdapter handles the wiring, order, and data flow for you.


