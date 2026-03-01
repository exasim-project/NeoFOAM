# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Unit tests for BaseSpec — shared base class for ModelSpec and SolverSpec."""

from types import SimpleNamespace
from typing import Any

from neofoam.io import BaseConfig
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.operations import Operation


# ===========================================================================
# Cycle 1 — BaseSpec.__init__ + config() decorator
# ===========================================================================


def test_base_spec_stores_name() -> None:
    """BaseSpec stores its name."""
    from neofoam.framework.base_spec import BaseSpec

    spec = BaseSpec("test")
    assert spec.name == "test"


def test_base_spec_config_decorator_stores_class() -> None:
    """@spec.config registers the config class."""
    from neofoam.framework.base_spec import BaseSpec

    spec = BaseSpec("test")

    @spec.config
    class MyCfg(BaseConfig):
        x: int = 1

    assert spec._config_class is MyCfg


def test_base_spec_has_dependency_resolver() -> None:
    """BaseSpec creates a DependencyResolver."""
    from neofoam.framework.base_spec import BaseSpec
    from neofoam.framework.operation_wrapper import DependencyResolver

    spec = BaseSpec("test")
    assert isinstance(spec._dependency_resolver, DependencyResolver)


# ===========================================================================
# Cycle 2 — operation() decorator
# ===========================================================================


def test_base_spec_operation_stores_func_and_metadata() -> None:
    """@spec.operation registers (func, metadata) in spec._operations."""
    from neofoam.framework.base_spec import BaseSpec

    spec = BaseSpec("test")

    @spec.operation(operation_number="1.0", name="my_step")
    def my_step(self: Any, x: float) -> None:
        pass

    assert len(spec._operations) == 1
    func, meta = spec._operations[0]
    assert func is my_step
    assert meta["operation_number"] == "1.0"
    assert meta["name"] == "my_step"


def test_base_spec_operation_default_name_from_func() -> None:
    """Operation name defaults to function name."""
    from neofoam.framework.base_spec import BaseSpec

    spec = BaseSpec("test")

    @spec.operation(operation_number="2.0")
    def compute(self: Any) -> None:
        pass

    _, meta = spec._operations[0]
    assert meta["name"] == "compute"


def test_base_spec_operation_stores_depends_on_and_before() -> None:
    """Operation metadata stores depends_on and before lists."""
    from neofoam.framework.base_spec import BaseSpec

    spec = BaseSpec("test")

    @spec.operation(
        operation_number="1.0",
        depends_on=["init"],
        before=["finalize"],
    )
    def step(self: Any) -> None:
        pass

    _, meta = spec._operations[0]
    assert meta["depends_on"] == ["init"]
    assert meta["before"] == ["finalize"]


# ===========================================================================
# Cycle 3 — _build_operations_for(runtime)
# ===========================================================================


class StubConfig(BaseConfig):
    factor: float = 2.0


def test_base_spec_build_operations_returns_operation_list() -> None:
    """_build_operations_for returns a list of Operation objects."""
    from neofoam.framework.base_spec import BaseSpec

    spec = BaseSpec("test")

    @spec.operation(operation_number="1.0")
    def my_step(self: Any, cfg: StubConfig) -> None:
        pass

    runtime = SimpleNamespace(config=StubConfig(factor=3.0))
    ops = spec._build_operations_for(runtime)

    assert len(ops) == 1
    assert isinstance(ops[0], Operation)
    assert ops[0].metadata.op_name == "my_step"
    assert str(ops[0].metadata.operation_number) == "1.0"


def test_base_spec_build_operations_wrapper_is_callable() -> None:
    """The wrapped operation can be called with a Context."""
    from neofoam.framework.base_spec import BaseSpec

    spec = BaseSpec("test")

    @spec.operation(operation_number="1.0")
    def add_result(self: Any, cfg: StubConfig, x: float) -> FieldUpdates:
        return FieldUpdates({"result": x * cfg.factor})

    runtime = SimpleNamespace(config=StubConfig(factor=2.0))
    ops = spec._build_operations_for(runtime)

    ctx = Context(fields={"x": 5.0}, models={})
    ops[0].run(ctx)
    assert ctx.fields["result"] == 10.0


def test_base_spec_build_operations_with_suffix() -> None:
    """_build_operations_for appends suffix to operation names."""
    from neofoam.framework.base_spec import BaseSpec

    spec = BaseSpec("test")

    @spec.operation(operation_number="1.0")
    def step(self: Any) -> None:
        pass

    runtime = SimpleNamespace(config=None)
    ops = spec._build_operations_for(runtime, suffix="_zoneA")

    assert ops[0].metadata.op_name == "step_zoneA"


def test_base_spec_build_operations_dep_resolution_path() -> None:
    """When no BaseConfig param, uses dependency resolution (field injection)."""
    from neofoam.framework.base_spec import BaseSpec

    spec = BaseSpec("test")

    @spec.operation(operation_number="1.0")
    def step(self: Any, x: float) -> FieldUpdates:
        return FieldUpdates({"y": x + 1})

    runtime = SimpleNamespace(config=None)
    ops = spec._build_operations_for(runtime)

    ctx = Context(fields={"x": 10.0}, models={})
    ops[0].run(ctx)
    assert ctx.fields["y"] == 11.0
