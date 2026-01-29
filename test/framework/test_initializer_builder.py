# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for InitializerBuilder fluent API."""

from foamadapter.framework.initialization import InitializerBuilder, LazyInit


def test_initializer_builder_empty():
    """Test empty builder returns empty list."""
    builder = InitializerBuilder()
    result = builder.build()

    assert len(result) == 0
    assert result == []


def test_initializer_builder_add_resource():
    """Test adding resources."""
    builder = InitializerBuilder()
    mesh_config = {"nPoints": 100, "name": "domain"}
    result = builder.add_resource("mesh", mesh_config).build()

    assert len(result) == 1
    assert result[0].name == "mesh"
    assert result[0].execute() == mesh_config


def test_initializer_builder_add_model_with_value():
    """Test adding models with constant values."""
    builder = InitializerBuilder()
    algorithm = {"param1": 1e-5}
    result = builder.add_model("algorithm", algorithm).build()

    assert len(result) == 1
    assert result[0].name == "models.algorithm"
    assert result[0].category == "models"
    assert result[0].execute() == algorithm


def test_initializer_builder_add_model_with_callable():
    """Test adding models with callable."""
    builder = InitializerBuilder()

    def create_algorithm(ctx):
        return {"created": True}

    result = builder.add_model("algorithm", create_algorithm).build()

    assert len(result) == 1
    assert result[0].name == "models.algorithm"
    assert result[0].initializer({}) == {"created": True}


def test_initializer_builder_add_field_constant():
    """Test adding fields with constant values."""
    builder = InitializerBuilder()
    result = builder.add_field("U", depends_on=["mesh"], value=1.0).build()

    assert len(result) == 1
    assert result[0].name == "fields.U"
    assert result[0].category == "fields"
    assert result[0].depends_on == ["mesh"]
    assert result[0].execute() == 1.0


def test_initializer_builder_add_field_computed():
    """Test adding fields with computed values."""
    builder = InitializerBuilder()

    def compute_pressure(ctx):
        return ctx["fields.U"] * 2

    result = builder.add_field(
        "p", depends_on=["fields.U"], value=compute_pressure
    ).build()

    assert len(result) == 1
    assert result[0].name == "fields.p"
    assert result[0].depends_on == ["fields.U"]

    # Test computation
    ctx = {"fields.U": 5.0}
    assert result[0].initializer(ctx) == 10.0


def test_initializer_builder_add_operator():
    """Test adding operators."""
    builder = InitializerBuilder()

    def create_momentum(ctx):
        return {"type": "momentum"}

    result = builder.add_operator(
        "momentum", depends_on=["fields.U", "fields.p"], value=create_momentum
    ).build()

    assert len(result) == 1
    assert result[0].name == "operators.momentum"
    assert result[0].category == "operators"
    assert result[0].depends_on == ["fields.U", "fields.p"]


def test_initializer_builder_chaining():
    """Test fluent API chaining."""
    builder = InitializerBuilder()
    mesh = {"nPoints": 100}
    algorithm = {"param": 1.0}

    result = (
        builder.add_resource("mesh", mesh)
        .add_resource("domain", {"name": "domain"})
        .add_model("algorithm", algorithm)
        .add_field("U", depends_on=["mesh"], value=1.0)
        .add_field("p", depends_on=["mesh"], value=lambda ctx: 101325.0)
        .build()
    )

    assert len(result) == 5
    assert result[0].name == "mesh"
    assert result[1].name == "domain"
    assert result[2].name == "models.algorithm"
    assert result[3].name == "fields.U"
    assert result[4].name == "fields.p"


def test_initializer_builder_add_preconstructed():
    """Test adding pre-constructed LazyInit objects."""
    builder = InitializerBuilder()
    custom_init = LazyInit(
        name="custom.object",
        depends_on=["mesh"],
        initializer=lambda ctx: {"custom": True},
        category="custom",
    )

    result = builder.add(custom_init).build()

    assert len(result) == 1
    assert result[0].name == "custom.object"
    assert result[0].category == "custom"


def test_initializer_builder_extend():
    """Test extending with multiple initializers."""
    builder = InitializerBuilder()

    extra_inits = [
        LazyInit(name="field1", initializer=lambda: 1.0),
        LazyInit(name="field2", initializer=lambda: 2.0),
    ]

    result = builder.add_resource("mesh", {}).extend(extra_inits).build()

    assert len(result) == 3
    assert result[0].name == "mesh"
    assert result[1].name == "field1"
    assert result[2].name == "field2"


def test_initializer_builder_complex_workflow():
    """Test a realistic initialization workflow."""
    mesh_config = {"nPoints": 1000}
    solver_config = {"dt": 0.01, "endTime": 1.0}

    builder = InitializerBuilder()
    result = (
        builder
        # Resources
        .add_resource("mesh", mesh_config)
        .add_resource("config", solver_config)
        # Models
        .add_model("algorithm", {"param1": 1e-5})
        .add_model("transport", lambda ctx: {"type": "singlePhase"})
        # Fields
        .add_field("U", depends_on=["mesh"], value=[0, 0, 0])
        .add_field("p", depends_on=["mesh"], value=101325.0)
        .add_field("nu", depends_on=["models.transport"], value=lambda ctx: 1e-6)
        # Operators
        .add_operator(
            "momentum",
            depends_on=["fields.U", "fields.p"],
            value=lambda ctx: {"equation": "ddt(U) + div(phi, U) = -grad(p)"},
        )
        .build()
    )

    assert len(result) == 8

    # Verify names
    names = [init.name for init in result]
    assert "mesh" in names
    assert "config" in names
    assert "models.algorithm" in names
    assert "models.transport" in names
    assert "fields.U" in names
    assert "fields.p" in names
    assert "fields.nu" in names
    assert "operators.momentum" in names


def test_initializer_builder_lambda_capture():
    """Test that lambda captures work correctly for multiple items."""
    builder = InitializerBuilder()

    values = [1.0, 2.0, 3.0]
    for i, val in enumerate(values):
        builder.add_field(f"field{i}", depends_on=["mesh"], value=val)

    result = builder.build()

    assert len(result) == 3
    assert result[0].execute() == 1.0
    assert result[1].execute() == 2.0
    assert result[2].execute() == 3.0


def test_initializer_builder_returns_self():
    """Test that all builder methods return self for chaining."""
    builder = InitializerBuilder()

    assert builder.add_resource("mesh", {}) is builder
    assert builder.add_model("algorithm", {}) is builder
    assert builder.add_field("U", depends_on=[], value=1.0) is builder
    assert builder.add_operator("momentum", depends_on=[], value=lambda: {}) is builder
    assert builder.add(LazyInit(name="test", initializer=lambda: None)) is builder
    assert builder.extend([]) is builder
