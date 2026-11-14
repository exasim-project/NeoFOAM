# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — keeps parity with the framework model modules.

"""NeoN write hook for the framework ``fieldWriter``: the VoF fields, no turbulence fields."""

from typing import Any, Literal, Mapping

from neofoam import neofoam_bindings as nfb
from neofoam.algorithms.field_writer.writer import FieldHook
from neofoam.framework.initialization import InitStep, model

__all__ = ["NeoNVoFWriteHook", "neon_vof_writer_backend_steps"]


@FieldHook.register
class NeoNVoFWriteHook(FieldHook):
    """Write the flagged NeoN fields by their Context name (``alpha1``, ``p_rgh``, ``p``, ``U``).

    Use it for the VoF fields; ``incompressibleFluidNeoN`` has its own hook. Each lands under
    its own field name (``alpha.water`` for ``alpha1``) in the time directory ``NeoNTimeSync``
    has already set on ``Foam::Time``. Call it from the time loop's thread only.

    Example::

        NeoNVoFWriteHook(runtime=neon_runtime).write_fields({"U": velocity})
    """

    field_hook_type: Literal["neonVoF"] = "neonVoF"
    runtime: Any = None  # the NeoN RunTime adapter
    scalar_names: tuple[str, ...] = ("alpha1", "p_rgh", "p")
    vector_names: tuple[str, ...] = ("U",)

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        """Write the fields named in ``scalar_names`` and ``vector_names``, ignoring the rest."""
        for name, field_values in fields.items():
            if name in self.scalar_names:
                nfb.write_scalar_field(field_values, self.runtime)
            elif name in self.vector_names:
                nfb.write_vector_field(field_values, self.runtime)


def neon_vof_writer_backend_steps() -> list[InitStep]:
    """Inject :class:`NeoNVoFWriteHook` and the per-step ``printExecutionTime`` reporter.

    Use it in the VoF solver's init; ``neon_writer_backend_steps`` is the single-phase
    alternative. The steps run once, after ``models.writer`` and the NeoN runtime exist.

    Example::

        builder.extend(neon_vof_writer_backend_steps())
    """

    def inject_hook(ctx: dict[str, Any]) -> FieldHook:
        hook = NeoNVoFWriteHook(runtime=ctx["_neon_runtime"])
        ctx["models.writer"].hook = hook
        return hook

    def make_reporter(ctx: dict[str, Any]) -> Any:
        registry = ctx["_foam_time"]
        return lambda: registry.printExecutionTime()

    return [
        model("writer_hook", inject_hook, depends_on=["models.writer", "_neon_runtime"]),
        model("step_reporter", make_reporter, depends_on=["_foam_time"]),
    ]
