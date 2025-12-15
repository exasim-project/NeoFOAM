Solver and Models
=================


A solver with it models is just a sequence of operations that are executed in a specific order.
The job of the solver is to define the operations and sequence of operations.
Additionally, the solver also defines the models that can be used to extend the solver functionality as sketched in the code snippet below.

.. code-block:: python

    @Solver
    class IncompressibleFluidSolver:
        models: list[IncompressibleFluidModel]  # Additional physics models
        @Solver.operation(...)
        def momentum(self, ...): pass
        @Solver.operation(...)
        def continuity(self, ...): pass
        @Solver.operation(...)
        def update_turbulence(self, ...): pass


        def define_operations(self):
            # ...
            return operations

    @IncompressibleFluidModel.register
    class BoussinesqModel:
        @IncompressibleFluidModel.operation(...)
        def temperature_equation(self, ...): pass

    @IncompressibleFluidModel.register
    class PorosityModel:
        @IncompressibleFluidModel.operation(...)
        def add_momentum_source(self, ...): pass

    @IncompressibleFluidModel.register
    class RotatingReferenceFrame:
        @IncompressibleFluidModel.operation(...)
        def add_momentum_source(self, ...): pass


The example provides an overview but it gets more complex if a more complex/detailed solver is considered that is ideally composed of reusable compoenents and models.

The PressureVelocityCoupling supports different algorithms: SIMPLE, PISO, PIMPLE and the projection method.
However, it also supports different strategies to solver the pressure equation in cases natural convection is present or not.
This means that the models can influence the configuration of the operations of each model or the solver itself.
This is solved by a 3 stage initialization process of the solver and models:

1. load: Load configuration and initialize models and solver
2. resolve_dependencies: Resolve and connect models based on the loaded configuration (also is used for interaction with other domains in a multiphysics simulation)
3. build: Build models and solver, allocate fields and data structures

.. code-block:: python

    @Solver
    class IncompressibleFluidSolver:
    #...
        @classmethod
        def read(cls):
            pU = PressureVelocityCoupling.read()
            turbulenceModel = TurbulenceModel.read()
            models = read_models(IncompressibleFluidModel)

            return cls(pU, turbulenceModel, models)

        def resolve_dependencies(self):
            # ... TBD

        def build(self):
            # ... TBD

        def operations(self):
            # ... TBD

        def define_operations(self, domain_name: str | None = None) -> Operations:
            # ... TBD
