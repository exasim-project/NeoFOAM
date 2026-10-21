# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``fallback`` operation tag and its ``ModelRuntime`` partition helpers.

An ``@spec.operation`` may be tagged ``fallback=True``; a runtime then hands
back its native vs fallback operations separately, while ``.operations`` still
returns both. This is the single framework seam the turbulence-family merge
relies on to co-locate a model's native and pybFoam-fallback ``correct`` ops in
one file and dispatch by a solver flag.
"""

from typing import Any

from neofoam.framework.model import ModelRuntime, ModelSpec


def _spec_with_native_and_fallback_ops() -> ModelSpec:
    spec = ModelSpec("M")

    @spec.operation(name="nativeStep")
    def native_step(self: Any, ctx: Any) -> None:
        pass

    @spec.operation(name="fallbackStep", fallback=True)
    def fallback_step(self: Any, ctx: Any) -> None:
        pass

    return spec


def test_native_operations_excludes_fallback_ops() -> None:
    rt = ModelRuntime(spec=_spec_with_native_and_fallback_ops(), name="M", config={})
    names = [op.metadata.op_name for op in rt.native_operations()]
    assert names == ["nativeStep"]


def test_fallback_operations_returns_only_fallback_ops() -> None:
    rt = ModelRuntime(spec=_spec_with_native_and_fallback_ops(), name="M", config={})
    names = [op.metadata.op_name for op in rt.fallback_operations()]
    assert names == ["fallbackStep"]


def test_operations_still_returns_both() -> None:
    rt = ModelRuntime(spec=_spec_with_native_and_fallback_ops(), name="M", config={})
    assert len(rt.operations) == 2


def test_operation_is_native_by_default() -> None:
    spec = ModelSpec("M")

    @spec.operation(name="plain")
    def plain(self: Any, ctx: Any) -> None:
        pass

    rt = ModelRuntime(spec=spec, name="M", config={})
    assert rt.fallback_operations() == []
    assert [op.metadata.op_name for op in rt.native_operations()] == ["plain"]
