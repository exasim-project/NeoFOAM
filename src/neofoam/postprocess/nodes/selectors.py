# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spatial selectors — narrow a dataset to the elements inside a region."""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import BeforeValidator, SerializeAsAny

from neofoam.postprocess.node import DataSet, Node


class Selector(Node):
    """Restrict the active elements to a region; combine with ``&``, ``|``, ``~``.

    The base of every region node: a subclass implements :meth:`select` on the
    positions alone and inherits the mask bookkeeping, so a selector never
    resurrects an element an upstream node masked out. An aggregator downstream
    honours the mask::

        field("p") | (Box(min=(0, 0, 0), max=(1, 1, 1)) & ~Sphere(center=(0, 0, 0), radius=0.1))
    """

    def select(self, positions: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """The elements of ``positions`` (shape ``(n, 3)``) inside the region."""
        raise NotImplementedError

    def compute(self, dataset: DataSet) -> DataSet:
        # AND with the inbound mask: a line source already masks points outside
        # the mesh, and a selector must not resurrect them.
        selected = self.select(np.asarray(dataset.geometry.positions))
        if dataset.mask is not None:
            selected = np.asarray(dataset.mask) & selected
        return dataset.with_mask(selected)

    def __and__(self, other: Selector) -> Binary:
        return Binary(op="and", left=self, right=other)

    def __or__(self, other: Selector) -> Binary:
        return Binary(op="or", left=self, right=other)

    def __invert__(self) -> Not:
        return Not(region=self)


def _resolve_selector(value: Any) -> Any:
    """Turn a nested ``{"type": ...}`` mapping into the selector it names."""
    if not isinstance(value, dict):
        return value
    node = cast(Any, Node).create(node=value).node
    if not isinstance(node, Selector):
        raise ValueError(f"expected a selector, {value.get('type')!r} is a {type(node).__name__}")
    return node


#: A selector held by another selector. A plain ``Selector`` annotation would
#: validate a mapping into an empty base instance, so resolve it here against
#: the ``Node`` union as it stands at validation time — a selector registered by
#: a case script is nestable too.
NestedSelector = Annotated[SerializeAsAny[Selector], BeforeValidator(_resolve_selector)]


@Node.register
class Box(Selector):
    """Elements inside an axis-aligned box, boundary included.

    The cheapest region; use :class:`Sphere` for a radial one::

        field("p") | Box(min=(0.0, 0.0, 0.0), max=(1.0, 1.0, 1.0)) | Mean()
    """

    type: Literal["box"] = "box"
    min: tuple[float, float, float]
    max: tuple[float, float, float]

    def select(self, positions: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.all((positions >= self.min) & (positions <= self.max), axis=1)


@Node.register
class Sphere(Selector):
    """Elements within ``radius`` of ``center``, boundary included.

    The radial counterpart of :class:`Box`::

        field("p") | Sphere(center=(0.0, 0.0, 0.0), radius=0.05) | Mean()
    """

    type: Literal["sphere"] = "sphere"
    center: tuple[float, float, float]
    radius: float

    def select(self, positions: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return cast(
            "np.ndarray[Any, Any]",
            np.linalg.norm(positions - np.asarray(self.center), axis=1) <= self.radius,
        )


@Node.register
class Not(Selector):
    """Everything the wrapped region does *not* select.

    Written ``~region`` in Python and ``{type: not, region: {...}}`` in a spec
    file; the inversion applies to the region alone, so elements masked out
    upstream stay out::

        field("p") | ~Box(min=(0.0, 0.0, 0.0), max=(1.0, 1.0, 1.0)) | Mean()
    """

    type: Literal["not"] = "not"
    region: NestedSelector

    def select(self, positions: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return ~self.region.select(positions)


@Node.register
class Binary(Selector):
    """The intersection (``and``) or union (``or``) of two regions.

    Written ``left & right`` / ``left | right`` in Python — note that ``|``
    between two selectors builds this node, while ``|`` after a pipeline appends
    a step::

        field("p") | (Box(min=(0, 0, 0), max=(1, 1, 1)) | Sphere(center=(2, 0, 0), radius=1))
    """

    type: Literal["binary"] = "binary"
    op: Literal["and", "or"]
    left: NestedSelector
    right: NestedSelector

    def select(self, positions: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        left = self.left.select(positions)
        right = self.right.select(positions)
        combined = left & right if self.op == "and" else left | right
        return cast("np.ndarray[Any, Any]", combined)
