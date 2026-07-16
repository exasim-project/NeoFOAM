# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The :class:`Sweep` object — one value that owns the parameter-sweep round-trip.

Instead of hand-wiring :func:`cross_product`, :func:`sweep_snakefile`, the
:class:`~neofoam.tooling.workflow.rules.RulePlan` resolution and the
``sweep.meta.json`` sidecar, a caller describes the sweep once and gets
:meth:`Sweep.export` / :meth:`Sweep.load` plus the derived views
(:attr:`~Sweep.rows`, :attr:`~Sweep.plan`, :attr:`~Sweep.snakefile`) off one object.
The free functions it delegates to remain importable (they are its helpers), but a
new caller only needs this class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from neofoam.tooling.workflow.rules import (
    CAD_DIM,
    RulePlan,
    RuleRegistry,
    default_registry,
)
from neofoam.tooling.workflow.sweep._codegen import sweep_snakefile
from neofoam.tooling.workflow.sweep._io import (
    LoadedSweep,
    SweepExport,
    cross_product,
    export_sweep,
    load_sweep,
    merge_cad,
    plan_enabled_with_cad,
)


@dataclass(frozen=True)
class Sweep:
    """A parameter sweep as a value: dimensions + solver + base case, round-trip owned.

    ``dimensions`` maps each swept axis to its named variants
    (``{axis: {variant: payload}}``); the reserved ``mesh`` axis is keyed. ``classes``
    supplies the pydantic config class per axis, used to validate variants on
    :meth:`export`. A CAD axis is passed as ``cad={"cad": {"model": path, "variants":
    {...}}}`` (opt-in; enables the ``cad_geometry`` rule).

    :meth:`export` writes the runnable workflow directory; :meth:`load` reads one back.
    :attr:`rows`, :attr:`plan` and :attr:`snakefile` are derived read-only views — the
    caller never wires the underlying free functions together.
    """

    dimensions: Mapping[str, dict[str, dict[str, Any]]]
    solver_name: str
    base_case: str | Path
    classes: Mapping[str, type[BaseModel]] = field(default_factory=dict)
    cad: Mapping[str, Mapping[str, Any]] | None = None
    registry: RuleRegistry | None = None
    enabled: Sequence[str] | None = None

    def export(self, out_dir: str | Path) -> SweepExport:
        """Write the runnable workflow dir (sweep.csv, params.yaml, Snakefile, configs)."""
        return export_sweep(
            out_dir,
            solver_name=self.solver_name,
            base_case=self.base_case,
            dimensions=self.dimensions,
            classes=self.classes,
            cad=self.cad,
            registry=self.registry,
            enabled=self.enabled,
        )

    @classmethod
    def load(cls, out_dir: str | Path) -> Sweep:
        """Read an exported sweep directory back into a :class:`Sweep`.

        The reconstructed sweep carries no config ``classes`` (they are not
        persisted), so it is for inspection (:attr:`rows`, :attr:`plan`,
        :attr:`snakefile`) rather than re-:meth:`export`.
        """
        loaded: LoadedSweep = load_sweep(out_dir)
        dims = {d: v for d, v in loaded.dimensions.items() if d != CAD_DIM}
        cad: dict[str, Mapping[str, Any]] | None = None
        if loaded.cad_model:
            cad = {
                CAD_DIM: {
                    "model": loaded.cad_model,
                    "variants": loaded.dimensions.get(CAD_DIM, {}),
                }
            }
        return cls(
            dimensions=dims,
            solver_name=loaded.solver_name,
            base_case=loaded.base_case,
            cad=cad,
            enabled=loaded.enabled,
        )

    # -- derived read-only views -------------------------------------------
    def _resolved(self) -> tuple[dict[str, dict[str, Any]], str | None]:
        """Effective dimensions (CAD axis folded in) and the CAD model path."""
        return merge_cad(self.dimensions, self.cad)

    @property
    def rows(self) -> list[dict[str, str]]:
        """The cross-product rows (``sweep.csv``), with the CAD axis folded in."""
        dims, _ = self._resolved()
        return cross_product(dims)

    @property
    def plan(self) -> RulePlan:
        """The resolved rule plan (the ``cad_geometry`` rule on when a CAD axis is present)."""
        _, cad_model = self._resolved()
        enabled = plan_enabled_with_cad(self.enabled, has_cad=cad_model is not None)
        return (self.registry or default_registry()).plan(enabled)

    @property
    def snakefile(self) -> str:
        """The generated Snakefile text (same content :meth:`export` writes)."""
        dims, cad_model = self._resolved()
        enabled = plan_enabled_with_cad(self.enabled, has_cad=cad_model is not None)
        return sweep_snakefile(
            self.solver_name,
            self.base_case,
            sorted(dims),
            cad_model=cad_model,
            registry=self.registry,
            enabled=enabled,
        )
