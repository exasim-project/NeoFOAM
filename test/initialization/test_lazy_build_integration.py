# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Integration tests for lazy BUILD stage with DAG resolution."""

import pytest

pytestmark = pytest.mark.skip(reason="Outdated - framework refactored (SolverInitializer removed)")

# import networkx as nx
# from pydantic import BaseModel, Field
#
# from foamadapter.framework.initialization import SolverInitializer
from foamadapter.framework.initialization.helpers import field, lazy, operator
from foamadapter.framework.model import Model
from foamadapter.framework.bkp_solver import Solver


def test_lazy_build_with_field_dependencies():
    """Test that fields are initialized in dependency order."""
    execution_order = []

    class TestModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "test_model"
        setup_complete: bool = False

        @Model.build
        def initialize(self, mesh):
            """Return lazy initializers with dependencies."""

            def create_base():
                execution_order.append("base_field")
                return "base_value"

            def create_derived(context):
                execution_order.append("derived_field")
                # Should have access to base_field
                assert "fields.base_field" in context
                return f"derived_from_{context['fields.base_field']}"

            def mark_complete(context):
                execution_order.append("complete")
                self.setup_complete = True
                return None

            return [
                field("base_field", create=create_base),
                field(
                    "derived_field",
                    depends_on=["fields.base_field"],
                    create=create_derived,
                ),
                lazy(
                    "_complete",
                    depends_on=["fields.base_field", "fields.derived_field"],
                    create=mark_complete,
                ),
            ]

    class TestSolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        model: TestModel = Field(default_factory=TestModel)

        def get_models(self):
            return [self.model]

        @Solver.build
        def setup(self, mesh):
            return []

    solver = TestSolver()
    initializer = SolverInitializer(solver)
    context = initializer.initialize(mesh=None)

    # Verify execution order
    assert execution_order == ["base_field", "derived_field", "complete"]

    # Verify fields in context
    assert context.fields["base_field"] == "base_value"
    assert context.fields["derived_field"] == "derived_from_base_value"

    # Verify model is marked complete
    assert solver.model.setup_complete


def test_lazy_build_with_operator_dependencies():
    """Test that operators depend on required fields."""
    execution_order = []

    class TestModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "test_model"

        @Model.build
        def initialize(self, mesh):
            """Return lazy initializers for fields and operators."""

            def create_field_a():
                execution_order.append("field_a")
                return "value_a"

            def create_field_b():
                execution_order.append("field_b")
                return "value_b"

            def create_matrix(context):
                execution_order.append("matrix")
                # Should have access to both fields
                assert "fields.field_a" in context
                assert "fields.field_b" in context
                return "matrix_value"

            return [
                field("field_a", create=create_field_a),
                field("field_b", create=create_field_b),
                operator(
                    "pressure_matrix",
                    depends_on=["fields.field_a", "fields.field_b"],
                    create=create_matrix,
                ),
            ]

    class TestSolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        model: TestModel = Field(default_factory=TestModel)

        def get_models(self):
            return [self.model]

        @Solver.build
        def setup(self, mesh):
            return []

    solver = TestSolver()
    initializer = SolverInitializer(solver)
    context = initializer.initialize(mesh=None)

    # Verify fields created before operator
    field_a_idx = execution_order.index("field_a")
    field_b_idx = execution_order.index("field_b")
    matrix_idx = execution_order.index("matrix")

    assert field_a_idx < matrix_idx
    assert field_b_idx < matrix_idx

    # Verify operator in context (operators are stored as models)
    assert context.models["pressure_matrix"] == "matrix_value"


def test_lazy_build_cyclic_dependency_detection():
    """Test that cyclic dependencies are detected and reported."""

    class TestModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "test_model"

        @Model.build
        def initialize(self, mesh):
            """Return lazy initializers with cyclic dependencies."""
            return [
                field("a", depends_on=["fields.b"], create=lambda ctx: "a"),
                field("b", depends_on=["fields.c"], create=lambda ctx: "b"),
                field("c", depends_on=["fields.a"], create=lambda ctx: "c"),
            ]

    class TestSolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        model: TestModel = Field(default_factory=TestModel)

        def get_models(self):
            return [self.model]

        @Solver.build
        def setup(self, mesh):
            return []

    solver = TestSolver()
    initializer = SolverInitializer(solver)

    # Should raise error about cyclic dependency
    with pytest.raises((ValueError, nx.NetworkXUnfeasible)):
        initializer.initialize(mesh=None)


def test_lazy_build_with_mesh_dependency():
    """Test that lazy initializers can depend on mesh."""
    execution_order = []

    class TestModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "test_model"

        @Model.build
        def initialize(self, mesh):
            """Return lazy initializers that use mesh."""

            def create_field_with_mesh(context):
                execution_order.append("field_with_mesh")
                # Access mesh from context
                mesh_obj = context.get("mesh")
                return f"field_on_{mesh_obj}"

            return [
                field("velocity", depends_on=["mesh"], create=create_field_with_mesh)
            ]

    class TestSolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        model: TestModel = Field(default_factory=TestModel)

        def get_models(self):
            return [self.model]

        @Solver.build
        def setup(self, mesh):
            """Provide mesh as lazy initializer."""

            def provide_mesh():
                execution_order.append("mesh")
                return "test_mesh"

            return [lazy("mesh", create=provide_mesh)]

    solver = TestSolver()
    initializer = SolverInitializer(solver)
    context = initializer.initialize(mesh=None)

    # Verify mesh created first
    assert execution_order == ["mesh", "field_with_mesh"]

    # Verify field has mesh reference
    assert context.fields["velocity"] == "field_on_test_mesh"


def test_lazy_build_multiple_models_dependencies():
    """Test lazy initialization across multiple models."""
    execution_order = []

    class TransportModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "transport"

        @Model.build
        def initialize(self, mesh):
            def create_transport(context):
                execution_order.append("transport")
                U = context["fields.U"]
                return f"transport_for_{U}"

            return [
                field(
                    "laminarTransport", depends_on=["fields.U"], create=create_transport
                )
            ]

    class TurbulenceModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "turbulence"

        @Model.build
        def initialize(self, mesh):
            def create_turbulence(context):
                execution_order.append("turbulence")
                transport = context["fields.laminarTransport"]
                return f"turbulence_with_{transport}"

            return [
                field(
                    "turbulence",
                    depends_on=["fields.laminarTransport"],
                    create=create_turbulence,
                )
            ]

    class TestSolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        transport: TransportModel = Field(default_factory=TransportModel)
        turbulence: TurbulenceModel = Field(default_factory=TurbulenceModel)

        def get_models(self):
            return [self.transport, self.turbulence]

        @Solver.build
        def setup(self, mesh):
            def create_velocity():
                execution_order.append("U")
                return "velocity_field"

            return [field("U", create=create_velocity)]

    solver = TestSolver()
    initializer = SolverInitializer(solver)
    context = initializer.initialize(mesh=None)

    # Verify correct order: U -> transport -> turbulence
    assert execution_order == ["U", "transport", "turbulence"]

    # Verify fields
    assert context.fields["U"] == "velocity_field"
    assert context.fields["laminarTransport"] == "transport_for_velocity_field"
    assert (
        context.fields["turbulence"] == "turbulence_with_transport_for_velocity_field"
    )


def test_lazy_build_parallel_initialization():
    """Test that independent initializers can be executed in any valid order."""
    execution_order = []

    class TestModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "test_model"

        @Model.build
        def initialize(self, mesh):
            """Return multiple independent initializers."""

            def create_a():
                execution_order.append("a")
                return "a"

            def create_b():
                execution_order.append("b")
                return "b"

            def create_c():
                execution_order.append("c")
                return "c"

            def create_combined(context):
                execution_order.append("combined")
                # Depends on all three
                return (
                    f"{context['fields.a']}_{context['fields.b']}_{context['fields.c']}"
                )

            return [
                field("a", create=create_a),
                field("b", create=create_b),
                field("c", create=create_c),
                field(
                    "combined",
                    depends_on=["fields.a", "fields.b", "fields.c"],
                    create=create_combined,
                ),
            ]

    class TestSolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        model: TestModel = Field(default_factory=TestModel)

        def get_models(self):
            return [self.model]

        @Solver.build
        def setup(self, mesh):
            return []

    solver = TestSolver()
    initializer = SolverInitializer(solver)
    context = initializer.initialize(mesh=None)

    # Verify a, b, c are all created before combined
    a_idx = execution_order.index("a")
    b_idx = execution_order.index("b")
    c_idx = execution_order.index("c")
    combined_idx = execution_order.index("combined")

    assert a_idx < combined_idx
    assert b_idx < combined_idx
    assert c_idx < combined_idx

    # Verify combined field
    assert context.fields["combined"] == "a_b_c"


def test_lazy_build_context_passing():
    """Test that context is properly passed to initializer functions."""

    class TestModel(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        name: str = "test_model"
        captured_context: dict = Field(default_factory=dict)

        @Model.build
        def initialize(self, mesh):
            def capture_context(context):
                # Store context for inspection
                self.captured_context = dict(context)
                return "value"

            return [
                field("first", create=lambda: "first_value"),
                field("second", depends_on=["fields.first"], create=capture_context),
            ]

    class TestSolver(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        model: TestModel = Field(default_factory=TestModel)

        def get_models(self):
            return [self.model]

        @Solver.build
        def setup(self, mesh):
            return [lazy("runtime", create=lambda: "runtime_value")]

    solver = TestSolver()
    initializer = SolverInitializer(solver)
    context = initializer.initialize(mesh=None)

    # Verify context had required fields
    assert "fields.first" in solver.model.captured_context
    assert solver.model.captured_context["fields.first"] == "first_value"
    # runtime is added by solver which runs after models, so it won't be in model's captured context
    # but it should be in the final context
    assert context.runTime == "runtime_value"
