# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pipe-composed case construction: a start state + ordered steps, materialized.

A case is described as a :class:`Pipeline` — a *start* (``empty()`` or
``from_template()``) composed with ordered *steps* via ``|`` — and nothing touches
disk until :meth:`Pipeline.at` runs the start then each step in order. A
:class:`Pipeline` is a plain value: materialize it any number of times to get
independent directories.

``|`` has exactly one meaning (compose a pipeline) and ``.build_at()`` exactly one
(materialize). Piping a step onto an already-materialized :class:`CaseDir` yields a
*new* pipeline whose start copies that case — so ``base | step`` forks, and you pick
the destination with ``.build_at()``:

    base = (from_template(src) | mesh()).build_at(tmp / "base")
    variant = (base | patch("system/controlDict", endTime=0.1)).build_at(tmp / "variant")
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    import numpy as np

#: Populates a (not-yet-existing) destination directory.
Start = Callable[[Path], None]


@dataclass(frozen=True)
class CaseDir:
    """A materialized case directory.

    A case-*construction* handle: compose more steps with ``|``, and read a field
    back with :meth:`read_field`.
    """

    path: Path

    def __or__(self, step: "Step") -> "Pipeline":
        """Fork: a new pipeline that copies this case, then applies *step*."""
        return Pipeline(_copy_from(self.path), (step,))

    def read_field(self, name: str, *, time: str = "latest") -> "np.ndarray[Any, Any]":
        """Read field *name*'s internal field back as a numpy array.

        Delegates to :func:`neofoam.tooling.casebuild.reader.read_field`, which runs
        the read in a fresh subprocess (constructing ``Foam::Time`` twice in one
        interpreter corrupts OpenFOAM global state). Imported lazily so ``pipeline.py``
        stays free of numpy/subprocess at module top — reading is a separate concern
        from building.
        """
        from neofoam.tooling.casebuild.reader import read_field

        return read_field(self, name, time=time)


#: A build step: mutates a materialized case in place.
Step = Callable[[CaseDir], None]


@dataclass(frozen=True)
class Pipeline:
    """A start state plus ordered steps; immutable until :meth:`at`."""

    start: Start
    steps: tuple[Step, ...] = ()

    def __or__(self, step: Step) -> "Pipeline":
        """Return a new pipeline with *step* appended."""
        return Pipeline(self.start, self.steps + (step,))

    def build_at(self, dest: Union[Path, str]) -> CaseDir:
        """Materialize into *dest*: run the start, then each step in order."""
        dest = Path(dest)
        self.start(dest)
        case = CaseDir(dest)
        for step in self.steps:
            step(case)
        return case


def _copy_from(src: Path) -> Start:
    def start(dest: Path) -> None:
        shutil.copytree(src, dest, dirs_exist_ok=True)

    return start


def empty() -> Pipeline:
    """Start from a bare skeleton — ``system/``, ``constant/``, ``0/`` and nothing else."""

    def start(dest: Path) -> None:
        for sub in ("system", "constant", "0"):
            (dest / sub).mkdir(parents=True, exist_ok=True)

    return Pipeline(start)


def from_template(src: Union[Path, str]) -> Pipeline:
    """Start by copying the case at *src*, restoring ``0.orig`` → ``0`` if present."""
    source = Path(src)

    def start(dest: Path) -> None:
        shutil.copytree(source, dest, dirs_exist_ok=True)
        orig = dest / "0.orig"
        if orig.is_dir():
            zero = dest / "0"
            if zero.exists():
                shutil.rmtree(zero)
            shutil.copytree(orig, zero)

    return Pipeline(start)


def pipe(head: Union[Pipeline, CaseDir], *steps: Step) -> Pipeline:
    """Compose *steps* onto *head* (a pipeline, or a case to fork). ``|`` without operators."""
    base = head if isinstance(head, Pipeline) else Pipeline(_copy_from(head.path))
    for step in steps:
        base = base | step
    return base
