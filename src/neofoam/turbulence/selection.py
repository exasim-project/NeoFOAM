# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Turbulence model selection / factory.

Resolves the active turbulence model name from a turbulenceProperties-like config
and builds the momentum-transport handle a solver consumes. One family
(:class:`~neofoam.turbulence.momentumTransport.momentumTransportModel`) serves
both solvers; a single ``fallback`` flag decides how the selected model is built:

* ``fallback=False`` (``incompressibleFluidNeoN``) → the **native NeoN**
  ModelSpec: its ``@build`` seeds NeoN ``nut``/``nuEff`` and its native
  ``@operation``s solve the transport PDEs. Returns a
  :class:`~neofoam.turbulence.native.NeoNHandle`.
* ``fallback=True`` (``incompressibleFluid``) → a **pybFoam-OpenFOAM** wrapper
  that only advances a ``correct()`` op and delegates the momentum stress to
  pybFoam's own ``divDevReff``. Returns a
  :class:`~neofoam.turbulence.fallback.FallbackHandle`.

A model's shape falls out of which operations it declares: a native-only model
raises on ``fallback=True``; a fallback-only model (no ``@build``, only a
``fallback=True`` op — e.g. ``realizableKE``) raises on ``fallback=False``.

A name with **no** registered spec is not an error on the fallback path: OpenFOAM's
own run-time selection table can build any of its incompressible models, so the
selector hands the name to :class:`OpenFOAMTurbulenceModel` and says so on stdout
(the run is then honestly attributed to pybFoam, not to a NeoFOAM closure). A
*registered* name is still checked against the case's family — a RAS closure is
never built for an ``LES { LESModel … }`` entry.

:func:`select_turbulence_model` operates on a *duck-typed* config (any object
exposing ``simulationType`` and optional ``RAS`` / ``LES`` sub-objects).
:func:`select_from_case`, which loads the real OpenFOAM dictionary, is the
pybFoam-bound entry point.
"""

from pathlib import Path
from typing import Any, Literal, Optional, Union, cast, overload

from pybFoam import Pstream

from .config import TurbulencePropertiesConfig
from .fallback import FallbackHandle, OpenFOAMTurbulenceModel, TurbulenceFactory
from .momentumTransport import TurbulenceFamily, momentumTransportModel
from .native import NeoNHandle

__all__ = ["model_name", "select_turbulence_model", "select_from_case"]

SelectedModel = Union[NeoNHandle, FallbackHandle]


def model_name(config: Any) -> Optional[str]:
    """Resolve the active model name from a turbulenceProperties config.

    Returns ``"laminar"`` for ``simulationType laminar``, the ``RASModel`` for
    ``RAS``, the ``LESModel`` for ``LES``, or ``None`` when it cannot be
    determined (unknown ``simulationType`` or a missing sub-dictionary).
    """
    sim_type = getattr(config, "simulationType", None)
    if sim_type == "laminar":
        return "laminar"
    if sim_type == "RAS":
        ras = getattr(config, "RAS", None)
        return getattr(ras, "RASModel", None) if ras is not None else None
    if sim_type == "LES":
        les = getattr(config, "LES", None)
        return getattr(les, "LESModel", None) if les is not None else None
    return None


def _config_family(config: Any) -> Optional[TurbulenceFamily]:
    """The family (``laminar``/``RAS``/``LES``) the config's simulationType selects."""
    sim_type = getattr(config, "simulationType", None)
    if sim_type in ("laminar", "RAS", "LES"):
        return cast(TurbulenceFamily, sim_type)
    return None


@overload
def select_turbulence_model(
    config: Any,
    *,
    fallback: Literal[False],
    runtime: Any = ...,
    case_dir: Union[str, Path] = ...,
    nu: Any = ...,
    U: Any = ...,
    phi: Any = ...,
    transport: Any = ...,
    of_factory: Optional[TurbulenceFactory] = ...,
) -> NeoNHandle: ...


@overload
def select_turbulence_model(
    config: Any,
    *,
    fallback: Literal[True],
    runtime: Any = ...,
    case_dir: Union[str, Path] = ...,
    nu: Any = ...,
    U: Any = ...,
    phi: Any = ...,
    transport: Any = ...,
    of_factory: Optional[TurbulenceFactory] = ...,
) -> FallbackHandle: ...


@overload
def select_turbulence_model(
    config: Any,
    *,
    fallback: bool,
    runtime: Any = ...,
    case_dir: Union[str, Path] = ...,
    nu: Any = ...,
    U: Any = ...,
    phi: Any = ...,
    transport: Any = ...,
    of_factory: Optional[TurbulenceFactory] = ...,
) -> SelectedModel: ...


def select_turbulence_model(
    config: Any,
    *,
    fallback: bool,
    runtime: Any = None,
    case_dir: Union[str, Path] = ".",
    nu: Any = None,
    U: Any = None,
    phi: Any = None,
    transport: Any = None,
    of_factory: Optional[TurbulenceFactory] = None,
) -> SelectedModel:
    """Select and build the momentum-transport handle for the configured model.

    Args:
        config: A turbulenceProperties-like config (``simulationType`` + optional
            ``RAS``/``LES``).
        fallback: ``False`` → native NeoN path (needs ``runtime``/``nu``);
            ``True`` → pybFoam-OpenFOAM path (needs ``U``/``phi``/``transport``).
        runtime: The NeoN runtime (native path only).
        case_dir: Case directory the spec loads its config / builds against.
        nu: The Context molecular viscosity (native path only).
        U, phi, transport: pybFoam fields/transport (fallback path only).
        of_factory: Override for the pybFoam turbulence factory (tests).

    Raises:
        ValueError: if the config names no model, if the registered model's
            family (RAS/LES) differs from the case's ``simulationType``, or if
            the model does not support the requested backend.
    """
    name = model_name(config)
    if name is None:
        raise ValueError(
            "cannot resolve the turbulence model: simulationType must be "
            "laminar / RAS / LES with its matching sub-dictionary"
        )

    spec = momentumTransportModel.find_spec(name)
    if spec is None:
        if not fallback:
            raise ValueError(
                f"no turbulence model registered for {name!r}; the native NeoN path "
                "needs a registered closure — run it on incompressibleFluid (the "
                "pybFoam fallback) or port it"
            )
        # OpenFOAM's own run-time selection table builds it; say so, so a matching
        # run is attributed to pybFoam rather than to a NeoFOAM closure. Master-only,
        # like the OpenFOAM ``Info`` lines it sits between in the solver log.
        if Pstream.master():
            print(
                f"Turbulence model {name!r} has no NeoFOAM closure — "
                "running it on the pybFoam OpenFOAM fallback"
            )
        of = OpenFOAMTurbulenceModel(U, phi, transport, factory=of_factory)
        return FallbackHandle(of, of.operations)

    registered_family = momentumTransportModel.family_of(name)
    case_family = _config_family(config)
    if registered_family != case_family:
        raise ValueError(
            f"turbulence model {name!r} is registered as a {registered_family} closure "
            f"but the case selects it as {case_family} — OpenFOAM keeps a separate "
            f"selection table per family, so this is not the same model"
        )

    model_runtime = spec.instantiate(Path(case_dir))

    if fallback:
        if not model_runtime.fallback_operations():
            raise ValueError(
                f"{name!r} declares no pybFoam fallback correct() — it has no "
                "fallback backend; run it on incompressibleFluidNeoN (native NeoN)"
            )
        of = OpenFOAMTurbulenceModel(U, phi, transport, factory=of_factory)
        return FallbackHandle(of, model_runtime.fallback_operations())

    native_capable = spec._build_func is not None or bool(model_runtime.native_operations())
    if not native_capable:
        raise ValueError(
            f"{name!r} has no native NeoN closure; run it on incompressibleFluid "
            "(pybFoam) or port it"
        )
    return NeoNHandle(model_runtime, runtime, nu)


def select_from_case(
    case_dir: Union[str, Path] = ".",
    *,
    fallback: bool,
    runtime: Any = None,
    nu: Any = None,
    U: Any = None,
    phi: Any = None,
    transport: Any = None,
    of_factory: Optional[TurbulenceFactory] = None,
) -> SelectedModel:
    """Load ``constant/turbulenceProperties`` from a case and select the model.

    This is the pybFoam-bound entry point: it imports and uses the OpenFOAM
    reading strategy via :class:`TurbulencePropertiesConfig`.
    """
    config = TurbulencePropertiesConfig.load(case_dir=case_dir)
    return select_turbulence_model(
        config,
        fallback=fallback,
        runtime=runtime,
        case_dir=case_dir,
        nu=nu,
        U=U,
        phi=phi,
        transport=transport,
        of_factory=of_factory,
    )
