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

:func:`select_turbulence_model` operates on a *duck-typed* config (any object
exposing ``simulationType`` and optional ``RAS`` / ``LES`` sub-objects).
:func:`select_from_case`, which loads the real OpenFOAM dictionary, is the
pybFoam-bound entry point.
"""

from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

from .fallback import FallbackHandle, OpenFOAMTurbulenceModel, TurbulenceFactory
from .momentumTransport import momentumTransportModel
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
        ValueError: if no model is registered for the configured name, or the
            model does not support the requested backend.
    """
    name = model_name(config)
    spec = momentumTransportModel.find_spec(name) if name is not None else None
    if spec is None:
        raise ValueError(f"no turbulence model registered for {name!r}")

    model_runtime = spec.instantiate(Path(case_dir))

    if fallback:
        if not model_runtime.fallback_operations():
            raise ValueError(
                f"{name!r} declares no pybFoam fallback correct() — it has no "
                "fallback backend; run it on incompressibleFluidNeoN (native NeoN)"
            )
        of = OpenFOAMTurbulenceModel(U, phi, transport, factory=of_factory)
        return FallbackHandle(of, model_runtime.fallback_operations())

    native_capable = spec._build_func is not None or bool(
        model_runtime.native_operations()
    )
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
    from .config import TurbulencePropertiesConfig

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
