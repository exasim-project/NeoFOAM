# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The native-NeoN momentum-transport handle (:class:`NeoNHandle`).

Returned by :func:`~neofoam.turbulence.selection.select_turbulence_model` when a
solver selects ``fallback=False`` (today: ``incompressibleFluidNeoN``). It is the
read/advance interface over a native model's :class:`ModelRuntime`, built on the
NeoN backend:

* :meth:`validate` runs the spec's ``@build`` InitSteps — seeding the NeoN
  ``runtime`` / ``nu`` / ``U`` the closures read — into a
  :class:`~neofoam.framework.context.Context` that owns the model's ``nut`` /
  ``nuEff`` (and, for a closure, its ``k`` / ``epsilon`` and helper operators).
* :meth:`correct` steps the spec's **native** ``@operation``s (a transport solve
  for a closure; nothing for ``laminar``). The model's co-located ``fallback=True``
  op is deliberately **not** stepped here — that belongs to the pybFoam path.

Imports the NeoN bindings at module top, so it is imported by NeoN code/tests
(the ``incompressibleFluidNeoN`` create_fields, the parity worker), not from the
turbulence package ``__init__``.
"""

from typing import Any, Optional

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.framework.context import Context
from neofoam.framework.initialization import field as init_field
from neofoam.framework.initialization import model as init_model
from neofoam.framework.initialization.execution import execute_initialization
from neofoam.framework.model import ModelRuntime

__all__ = ["NeoNHandle"]


class NeoNHandle:
    """Read/advance interface over a native NeoN momentum-transport ``ModelRuntime``.

    :meth:`validate` runs the spec's ``@build`` InitSteps — seeding the NeoN
    ``runtime`` / ``nu`` / ``U`` the closures read — into a
    :class:`~neofoam.framework.context.Context` that owns the model's ``nut`` /
    ``nuEff`` (and, for a closure, its ``k`` / ``epsilon`` and helper operators).
    :meth:`correct` then steps the spec's native ``@operation``s (a transport solve
    for a closure; nothing for ``laminar``) over that same Context.

    The accessor surface (:meth:`nut` / :meth:`nu_eff` / :meth:`rotate_old_times` /
    :meth:`write`) matches the C++ ``nfb.create_turbulence_model`` handle, so the
    incompressibleFluidNeoN solver consumes either interchangeably.
    """

    #: Volume fields a model may own that OpenFOAM auto-writes alongside p/U —
    #: persisted by :meth:`write` when present (``nuEff`` is a surface field and
    #: ``G`` a per-step temporary, so neither is written).
    _WRITE_FIELDS = ("nut", "k", "epsilon", "nuTilda", "omega")

    def __init__(self, runtime: ModelRuntime, neon_runtime: Any, nu: Any) -> None:
        self._runtime = runtime
        self._neon_runtime = neon_runtime
        self._nu = nu
        self._ctx: Optional[Context] = None

    def has_nut(self) -> bool:
        """Native NeoN closures always maintain a ``nut`` field (zero for laminar)."""
        return True

    def validate(self, U: Any) -> None:
        """Build the model's fields (``nut`` / ``nuEff`` …) from its ``@build`` steps.

        Seeds the NeoN ``runtime`` / ``nu`` / ``U`` the build closures inject by
        name (``models.neon_runtime`` / ``models.nu_vol`` / ``fields.U``, matching
        the incompressibleFluidNeoN init graph), then executes the spec's InitSteps
        in dependency order into a live :class:`Context`.
        """
        seed = [
            init_model("neon_runtime", lambda _ctx: self._neon_runtime),
            init_model("nu_vol", lambda _ctx: self._nu),
            init_field("U", lambda _ctx: U),
        ]
        self._ctx = execute_initialization(seed + self._runtime.run_build())

    def correct(self, U: Any, phi: Any, runtime: Any) -> None:
        """Advance the model one step by stepping its **native** ``@operation``s.

        A closure (kEpsilon, …) solves its transport PDEs and recomputes ``nut`` /
        ``nuEff``; ``laminar`` declares no native operations, so this is a no-op.
        The model's co-located ``fallback=True`` op belongs to the pybFoam path and
        is intentionally excluded here. ``U`` / ``phi`` are refreshed on the Context
        the operations read from.
        """
        if self._ctx is None:
            raise RuntimeError("validate() must be called before correct()")
        self._ctx.fields["U"] = U
        self._ctx.fields["phi"] = phi
        for op in self._runtime.native_operations():
            op.run(self._ctx)

    def field(self, name: str) -> Any:
        """Return the NeoN field the model owns under ``name`` (``nut`` / ``nuEff`` …)."""
        if self._ctx is None:
            raise RuntimeError("validate() must be called before field()")
        if name not in self._ctx.fields:
            raise KeyError(
                f"NeoN turbulence exposes no field {name!r} "
                f"(known: {sorted(self._ctx.fields)})"
            )
        return self._ctx.fields[name]

    def nut(self) -> Any:
        """The eddy viscosity ``nut`` (volume field) the model maintains."""
        return self.field("nut")

    def nu_eff(self) -> Any:
        """The effective viscosity ``nuEff`` (surface field) the model maintains."""
        return self.field("nuEff")

    def rotate_old_times(self) -> None:
        """No-op: each transport ``@operation`` rotates its own field before its solve.

        The C++ handle rotates its transport fields' old times at the start of a
        time step; the pure-Python closures do the equivalent ``nn.rotate_old_times``
        inside :meth:`correct` (nothing touches those fields in between).
        """

    def write(self, mesh: Any = None) -> None:
        """Write the owned volume fields (``nut`` + transport unknowns) to disk.

        Signature-compatible with the C++ handle's ``write(mesh)``; ``mesh`` is
        unused — the NeoN writer resolves the output through the runtime adapter.
        """
        if self._ctx is None:
            raise RuntimeError("validate() must be called before write()")
        for name in self._WRITE_FIELDS:
            if name in self._ctx.fields:
                nfb.write_scalar_field(self._ctx.fields[name], self._neon_runtime)
