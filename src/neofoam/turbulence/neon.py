# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The turbulence ModelSpec approach on the **NeoN** backend.

Runtime-selectable turbulence for NeoN, driven by the OpenFOAM
``constant/turbulenceProperties`` dict — the symmetric NeoN counterpart of the
pybFoam :func:`~neofoam.turbulence.selection.select_turbulence_model` /
:class:`~neofoam.turbulence.momentumTransport.momentumTransportModel`.

* :class:`neonMomentumTransportModel` is the plugin family of **pure-Python NeoN**
  ModelSpec models, the NeoN mirror of ``momentumTransportModel``. Each model is a
  :class:`~neofoam.framework.model.ModelSpec` (``Model("name") + config + @build +
  @operation``, like :mod:`neofoam.turbulence.models.laminar`) registered via
  ``register_with``; the bundled ones live under :mod:`neofoam.turbulence.models`
  (``neon_laminar`` / ``neon_kEpsilon``) and self-register on import.
* :class:`NeoNMomentumTransport` is the small read interface over a model's
  :class:`~neofoam.framework.model.ModelRuntime`, the NeoN mirror of
  ``SpecMomentumTransport``: it runs the spec's ``@build`` InitSteps (seeding the
  NeoN ``runtime`` / ``nu`` / ``U``) into a :class:`~neofoam.framework.context.Context`,
  then steps its ``@operation``s for the per-step ``correct``.
* :func:`build_neon_turbulence` resolves the configured name and builds the
  registered pure-Python model through the full ModelSpec lifecycle (there is no
  C++ fallback — the NeoN turbulence subsystem is pure-Python).

Imports the NeoN bindings at module top (like the incompressibleFluidNeoN solver),
so it is imported directly by NeoN code/tests, not from the turbulence package
``__init__`` (which must stay importable without a NeoN build).
"""

from typing import Any, Optional

from pydantic import BaseModel

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.context import Context
from neofoam.framework.initialization import field as init_field
from neofoam.framework.initialization import model as init_model
from neofoam.framework.initialization.execution import execute_initialization
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

from .selection import model_name

__all__ = [
    "neonMomentumTransportModel",
    "NeoNMomentumTransport",
    "build_neon_turbulence",
    "Model",
    "ModelRuntime",
    "ModelSpec",
]


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class neonMomentumTransportModel(BaseModel):
    """Plugin family of pure-Python NeoN momentum-transport models.

    The NeoN mirror of :class:`~neofoam.turbulence.momentumTransport.momentumTransportModel`,
    holding only the ModelSpec-based NeoN closures — **no C++ fallback**, so once
    every model is ported the legacy C++ path in :func:`build_neon_turbulence`
    deletes cleanly. Models register via ``Model("name").register_with(neonMomentumTransportModel)``.
    """

    @classmethod
    def all_specs(cls) -> list[ModelSpec]:
        """Return every registered NeoN momentum-transport spec (no detection)."""
        registry = PluginSystem.get_registered("neonMomentumTransportModel")
        if not registry:
            return []
        return [
            plugin_cls.get_model_instance(plugin_cls)
            for plugin_cls in registry.plugin_registry
            if hasattr(plugin_cls, "get_model_instance")
        ]

    @classmethod
    def registered_names(cls) -> list[str]:
        """Return the ``spec.name`` of every registered NeoN model."""
        return [spec.name for spec in cls.all_specs()]

    @classmethod
    def find_spec(cls, name: str) -> Optional[ModelSpec]:
        """Return the registered spec whose ``name`` matches, else ``None``."""
        for spec in cls.all_specs():
            if spec.name == name:
                return spec
        return None


class NeoNMomentumTransport:
    """Read interface over a NeoN momentum-transport :class:`ModelRuntime`.

    The NeoN mirror of :class:`~neofoam.turbulence.momentumTransport.SpecMomentumTransport`.
    :meth:`validate` runs the spec's ``@build`` InitSteps — seeding the NeoN
    ``runtime`` / ``nu`` / ``U`` the closures read — into a
    :class:`~neofoam.framework.context.Context` that owns the model's ``nut`` /
    ``nuEff`` (and, for a closure, its ``k`` / ``epsilon`` and helper operators).
    :meth:`correct` then steps the spec's ``@operation``s (a transport solve for a
    closure; nothing for ``laminar``) over that same Context.

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
        """Advance the model one step by stepping its ``@operation``s.

        A closure (kEpsilon, …) solves its transport PDEs and recomputes ``nut`` /
        ``nuEff``; ``laminar`` declares no operations, so this is a no-op. ``U`` /
        ``phi`` are refreshed on the Context the operations read from.
        """
        if self._ctx is None:
            raise RuntimeError("validate() must be called before correct()")
        self._ctx.fields["U"] = U
        self._ctx.fields["phi"] = phi
        for op in self._runtime.operations:
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


def build_neon_turbulence(
    config: Any, runtime: Any, nu: Any, case_dir: Any
) -> NeoNMomentumTransport:
    """Resolve the turbulence model from the OpenFOAM dict and build it on NeoN.

    Runtime-selectable, driven by ``constant/turbulenceProperties`` (via
    :func:`~neofoam.turbulence.selection.model_name` on the passed, validated
    ``config``) — mirrors the solver's ``create_fields`` turbulence wiring. The name
    must be registered in :class:`neonMomentumTransportModel`; the model is built
    through the full ModelSpec lifecycle (``instantiate`` → ``run_build`` →
    operations). Raises when the dict names no pure-Python NeoN model.
    """
    name = model_name(config)
    if name is None:
        raise ValueError("could not resolve a turbulence model from the config")
    spec = neonMomentumTransportModel.find_spec(name)
    if spec is None:
        raise ValueError(
            f"no pure-Python NeoN turbulence model registered for {name!r}"
        )
    return NeoNMomentumTransport(spec.instantiate(case_dir), runtime, nu)


# Import side-effect: register the bundled pure-Python NeoN models with
# neonMomentumTransportModel. Kept at the bottom so the family and Model re-export
# above are already bound when the model modules import them (mirrors how
# momentumTransport + models/laminar resolve).
from .models import neon_kEpsilon  # noqa: E402,F401
from .models import neon_kOmegaSST  # noqa: E402,F401
from .models import neon_laminar  # noqa: E402,F401
from .models import neon_spalartAllmaras  # noqa: E402,F401
